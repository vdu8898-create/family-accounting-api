import os
import json
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dashscope import Generation
from dotenv import load_dotenv
from datetime import datetime, timedelta

# 加载环境变量
load_dotenv()

app = FastAPI(title="Family Accounting AI Backend")

# 🔥 关键：允许前端 Vue 项目跨域调用
# 在生产环境中，建议将 "*" 替换为你前端的实际域名，如 "https://your-vue-app.com"
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 定义数据模型
class TransactionData(BaseModel):
    amount: float
    type: str
    member_name: Optional[str] = None
    is_income: bool
    transaction_date: Optional[str] = None
    note: Optional[str] = None

class ParseRequest(BaseModel):
    text: str
    # 可选：前端可以把当前的成员列表和分类列表传过来，提高准确率
    known_members: Optional[List[str]] = []
    known_categories: Optional[List[str]] = []

class ParseResponse(BaseModel):
    success: bool
    data: Optional[TransactionData]
    message: str

class MultiParseResponse(BaseModel):
    success: bool
    data: List[TransactionData]
    message: str

SYSTEM_PROMPT_BASE = """
你是一个专业的家庭记账助手。请从用户文本中提取信息并返回纯 JSON。
规则：
1. amount: 数字。
2. type: 必须从已知分类中选择，若无匹配则选"其他"。
3. member_name: 必须从已知成员中选择，若未提及或不在列表中则为 null。
4. is_income: true(收入) 或 false(支出)。
5. transaction_date: YYYY-MM-DD 格式。相对时间需转换为绝对日期（基准日：{today}）。
6. note: 简短备注。
只返回 JSON，无 Markdown 标记。
"""

SYSTEM_PROMPT_MULTI = """
你是一个专业的家庭记账助手。用户的输入可能包含**一笔或多笔**交易。
你的任务是识别所有的交易，并以 JSON 列表 (List) 的形式返回。
规则：
1. 如果只有一笔交易，列表长度为 1。
2. 如果有多个动作（如“买了A又买了B”），请拆分为列表中的多个对象。
3. 每个对象的字段定义：
   - amount: 数字
   - type: 从已知分类选
   - member_name: 从已知成员选，无则为 null，我就是爸爸，大林子就是大宝，畅畅就是小宝，亲爱的就是妈妈
   - is_income: true/false
   - transaction_date: YYYY-MM-DD(基准日期: {today})
   - note: 简短备注
4. **重要**：只返回纯 JSON 列表，例如[{{"amount": 10, "type": "餐饮", "is_income": false}}, {{"amount": 20, "type": "交通", "is_income": false}}]。不要包含任何解释文字或 Markdown 标记。
5. 如果用户输入包含多笔交易，请只提取第一笔，并在返回的 note 字段末尾追加 '(注：检测到多笔交易，仅解析第一笔)'。
"""

@app.post("/api/ai/parse", response_model=MultiParseResponse)
async def parse_transaction(request: ParseRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="文本不能为空")

    # 动态构建 Prompt，注入已知的成员和分类，防止 AI 瞎编
    members_str = ", ".join(request.known_members) if request.known_members else "未知"
    categories_str = ", ".join(request.known_categories) if request.known_categories else "餐饮, 交通, 购物, 娱乐, 医疗, 教育, 工资, 理财, 转账, 其他"
    
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    system_prompt = SYSTEM_PROMPT_MULTI.format(today=today_str)
    system_prompt += f"\n已知成员列表: [{members_str}]"
    system_prompt += f"\n已知分类列表: [{categories_str}]"

    try:
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': f"解析：'{request.text}'"}
        ]

        response = Generation.call(
            model='qwen-plus',
            messages=messages,
            result_format='message',
            temperature=0.1,
            api_key=os.getenv("DASHSCOPE_API_KEY")
        )

        if response.status_code == 200:
            content = response.output.choices[0].message.content
            # 清理 Markdown
            clean_json = content.replace('```json', '').replace('```', '').strip()
            
            try:
                # 有时候模型会返回单个对象而不是列表，做个兼容处理
                parsed_raw = json.loads(clean_json)
                if isinstance(parsed_raw, dict):
                    parsed_list = [parsed_raw] # 如果是单个对象，转为列表
                elif isinstance(parsed_raw, list):
                    parsed_list = parsed_raw
                else:
                    return MultiParseResponse(success=False, data=[], message="AI 返回格式既不是对象也不是列表")
                
                # 验证每一个对象
                validated_data = [TransactionData(**item) for item in parsed_list]
                return MultiParseResponse(success=True, data=validated_data, message=f"成功解析 {len(validated_data)} 笔")
            except Exception as e:
                return MultiParseResponse(success=False, data=[], message=f"JSON 解析失败: {str(e)}")
        else:
            return MultiParseResponse(success=False, data=[], message=f"AI 服务错误: {response.message}")

    except Exception as e:
        return MultiParseResponse(success=False, data=[], message=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("port", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)