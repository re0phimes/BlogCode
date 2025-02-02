import os
import json
import requests
from typing import Dict, List, Optional, Union, Literal, Any
from dotenv import load_dotenv
from DrissionPage import ChromiumPage, ChromiumOptions
import time
import subprocess
from pydantic import BaseModel, Field
from loguru import logger


load_dotenv()


# 基础响应模型
class ToolCallStatus(BaseModel):
    tool_call_status: Literal["success", "error"]
    error_message: Optional[str] = None


# 搜索结果模型
class SearchResult(BaseModel):
    title: str
    link: str


# 搜索元数据模型
class SearchMetadata(BaseModel):
    engine: str = Field(default="bing")
    query: str
    total_results: int
    timestamp: Optional[int] = None
    result_hash: Optional[int] = None


# Web搜索响应模型
class WebSearchResponse(ToolCallStatus):
    search_results: Optional[List[SearchResult]] = None
    metadata: Optional[SearchMetadata] = None


# function calling schema
class FunctionCallingSchema(BaseModel):
    name: str
    arguments: str


# 工具调用模型
class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: Dict[str, Union[str, Dict]]


# 工具响应模型
class ToolResponse(BaseModel):
    tool_call_id: str
    content: str


# 消息模型
class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None


def web_search_tool(query: str, search_engine: str = "bing") -> Dict:
    """
    使用DrissionPage进行网络搜索

    :param query: 搜索关键词
    :param search_engine: 搜索引擎，默认使用bing
    :return: 搜索结果
    """
    browser_process = None
    try:
        # 先启动 Chrome 浏览器进程
        logger.info("正在启动 Chrome 浏览器...")
        browser_process = subprocess.Popen(
            [
                "google-chrome",
                "--headless=new",
                "--remote-debugging-port=9222",
                "--no-sandbox",
                "--disable-gpu",
                "--disable-dev-shm-usage",
            ]
        )

        # 等待浏览器启动
        time.sleep(2)

        # 创建 ChromiumOptions 实例
        co = ChromiumOptions()
        co.set_argument("--headless=new")
        co.set_argument("--no-sandbox")
        co.set_argument("--disable-gpu")
        co.set_argument("--disable-dev-shm-usage")

        # 设置连接到已启动的浏览器
        co.set_local_port(9222)

        logger.info("正在连接到浏览器...")
        page = ChromiumPage(co)

        if search_engine.lower() == "bing":
            # 访问Bing并搜索
            logger.info(f"\n正在访问Bing搜索: {query}")
            page.get(f"https://www.bing.com/search?q={query}")
            time.sleep(2)  # 等待搜索结果加载

            # 使用xpath查找搜索结果
            logger.info("正在获取搜索结果...")
            search_items = page.eles('xpath://li[@class="b_algo"]')

            results = []
            for idx, item in enumerate(search_items):
                if idx >= 10:  # 只取前10个结果
                    break
                try:
                    # 获取标题、链接和描述
                    title = item.ele("tag:h2").text.strip()
                    link = item.ele("tag:a").link

                    result = {
                        "title": title,
                        "link": link,
                    }

                    # 流式输出每个搜索结果
                    logger.info(f"\n获取到第 {idx + 1} 个结果，标题: {title}, 链接: {link}")

                    results.append(result)

                except Exception as e:
                    logger.info(f"解析第 {idx + 1} 个结果时出错: {str(e)}")
                    continue

            if not results:
                return {
                    "tool_call_status": "error",
                    "error_message": "未找到有效的搜索结果",
                }

        return {
            "tool_call_status": "success",
            "search_results": results,
            "metadata": {
                "engine": "bing",
                "query": query,
                "total_results": len(results),
                # 添加时间戳和结果校验
                "timestamp": int(time.time()),
                "result_hash": hash(str(results)),
            },
        }

    except Exception as e:
        error_msg = str(e)
        logger.info(f"搜索过程中出错: {error_msg}")
        return {"tool_call_status": "error", "error_message": error_msg}

    finally:
        # 清理资源
        try:
            if "page" in locals():
                page.quit()
        except:
            pass

        try:
            if browser_process:
                logger.info("正在关闭浏览器进程...")
                browser_process.terminate()
                browser_process.wait(timeout=5)
        except:
            pass


def get_project_structure(
    root_dir: str, exclude_dirs: Optional[List[str]] = None
) -> Dict:
    """
    获取项目目录结构并返回结构化字典

    :param root_dir: 项目根目录路径
    :param exclude_dirs: 需要排除的目录列表
    :return: 包含目录结构的字典
    """
    # 设置默认排除目录
    if exclude_dirs is None:
        exclude_dirs = [
            "venv",
            ".venv",
            ".git",
            "__pycache__",
            ".idea",
            ".vscode",
            "node_modules",
            "dist",
            "build",
            ".mypy_cache",
            ".pytest_cache",
        ]

    # 转换为绝对路径并验证存在性
    root_dir = os.path.abspath(root_dir)
    if not os.path.exists(root_dir):
        raise ValueError(f"目录不存在：{root_dir}")

    def _traverse_directory(current_dir: str) -> Dict:
        structure = {
            "name": os.path.basename(current_dir),
            "type": "directory",
            "children": [],
        }

        try:
            entries = os.listdir(current_dir)
        except PermissionError:
            return structure

        # 按目录在前文件在后的顺序排序
        entries.sort(key=lambda x: (not os.path.isdir(os.path.join(current_dir, x)), x))

        for entry in entries:
            full_path = os.path.join(current_dir, entry)
            relative_path = os.path.relpath(full_path, root_dir)

            # 跳过排除目录
            if any(ex_dir in relative_path.split(os.sep) for ex_dir in exclude_dirs):
                continue

            if os.path.isdir(full_path):
                child = _traverse_directory(full_path)
                if (
                    child["children"] or not exclude_dirs
                ):  # 保留空目录如果不在排除列表中
                    structure["children"].append(child)
            else:
                structure["children"].append(
                    {"name": entry, "type": "file", "size": os.path.getsize(full_path)}
                )

        return structure

    try:
        return _traverse_directory(root_dir)
    except Exception as e:
        return {"error": str(e)}


def prepare_project_structure_tool(
    root_dir: str = ".", exclude_dirs: Optional[List[str]] = None
) -> Dict:
    """
    准备项目结构数据工具函数

    :param root_dir: 项目根目录（默认为当前目录）
    :param exclude_dirs: 需要排除的目录列表
    :return: 符合tools calling要求的格式
    """
    try:
        # 添加更多默认排除目录
        if exclude_dirs is None:
            exclude_dirs = [
                "venv",
                ".venv",
                ".git",
                "__pycache__",
                ".idea",
                ".vscode",
                "node_modules",
                "dist",
                "build",
                ".mypy_cache",
                ".pytest_cache",
                "target",
                "out",
                "output",
                "logs",
            ]

        structure = get_project_structure(root_dir, exclude_dirs)
        if "error" in structure:
            return {"tool_call_status": "error", "error_message": structure["error"]}

        # 简化返回的数据结构，只返回文件名和类型
        def simplify_structure(node):
            if node["type"] == "directory":
                return {
                    "name": node["name"],
                    "type": "directory",
                    "children": [
                        simplify_structure(child) for child in node["children"][:10]
                    ],  # 限制每个目录最多显示10个项目
                }
            return {"name": node["name"], "type": "file"}

        simplified_structure = simplify_structure(structure)

        return {
            "tool_call_status": "success",
            "structure_data": simplified_structure,
            "metadata": {"scanned_directory": os.path.abspath(root_dir)},
        }
    except Exception as e:
        return {"tool_call_status": "error", "error_message": str(e)}


# 只在处理用户消息时添加工具
tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "description": """用于获取实时信息或专业知识。适用场景：
1. 需要最新的新闻、数据或市场信息
2. 需要特定领域的专业技术细节
3. 需要验证某个说法或数据的准确性
4. 需要了解产品、技术或行业的最新发展

搜索结果将包含标题和链接，可用于进一步分析和参考。""",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "搜索关键词，应该精确描述所需信息",
                                },
                                "search_engine": {
                                    "type": "string",
                                    "description": "搜索引擎选择",
                                    "enum": ["bing"],
                                },
                            },
                            "required": ["query"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "prepare_project_structure",
                        "description": """用于分析项目的目录结构。适用场景：
1. 需要了解项目整体架构
2. 进行代码审查或技术评估
3. 提供项目改进建议
4. 解决项目相关的技术问题

返回项目的文件和目录层次结构，包含文件名和类型信息。""",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "root_dir": {
                                    "type": "string",
                                    "description": "项目根目录路径，使用相对或绝对路径",
                                },
                                "exclude_dirs": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "需要排除的目录列表，如node_modules、.git等",
                                },
                            },
                            "required": ["root_dir"],
                        },
                    },
                },
            ]



def send_messages(messages):
    """使用requests发送消息到API"""
    try:
        global tools
        api_key = os.getenv("DEEPSEEK_API_KEY")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # 检查最后一条消息是否为用户消息
        is_user_message = messages and messages[-1]["role"] == "user"

        data = {
            "model": "deepseek-chat",
            "messages": messages,
            "tools": tools if is_user_message else [],
            "tool_choice": "auto" if is_user_message else "none",
            "temperature": 0.7,
            "max_tokens": 4096,
        }

        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions", headers=headers, json=data
        )

        logger.debug(f"API调用结果: {response.json()}")

        if response.status_code != 200:
            return {
                "content": f"API调用失败: HTTP {response.status_code}, {response.text}"
            }

        result = response.json()
        return result["choices"][0]["message"]

    except Exception as e:
        return {"content": f"发生错误: {str(e)}"}


# 修改handle_tool_calls函数
def handle_tool_calls(message: Message) -> Optional[List[ToolResponse]]:
    """处理工具调用的函数"""
    if not message.tool_calls:
        return None

    results: List[ToolResponse] = []
    for tool_call in message.tool_calls:
        args = json.loads(tool_call.function["arguments"])
        tool_response = ToolResponse(tool_call_id=tool_call.id, content="")

        try:
            if tool_call.function["name"] == "web_search":
                result = web_search_tool(
                    query=args["query"], search_engine=args.get("search_engine", "bing")
                )
                web_search_response = WebSearchResponse(**result)
                tool_response.content = json.dumps(
                    web_search_response.model_dump(), ensure_ascii=False
                )
                results.append(tool_response)
            elif tool_call.function["name"] == "prepare_project_structure":
                result = prepare_project_structure_tool(
                    root_dir=args.get("root_dir", "."),
                    exclude_dirs=args.get("exclude_dirs", None),
                )
                tool_response.content = json.dumps(result, ensure_ascii=False)
                results.append(tool_response)

        except Exception as e:
            logger.info(f"执行工具 {tool_call.function['name']} 时出错: {str(e)}")
            error_response = ToolCallStatus(
                tool_call_status="error", error_message=str(e)
            )
            tool_response.content = json.dumps(
                error_response.model_dump(), ensure_ascii=False
            )
            results.append(tool_response)

    return results


# 默认的system prompt
default_system_message = {
    "role": "system",
    "content": """你是一个智能助手，擅长解决各类问题和完成各种任务。你具备以下能力：

1. 信息获取与分析
- 当需要最新信息、专业知识或验证信息时，可以使用web_search工具
- 对搜索结果进行分析整合，提取关键信息
- 结合已有知识和搜索结果给出全面的回答

2. 项目分析与理解
- 需要了解项目结构时，可使用prepare_project_structure工具
- 基于目录结构分析项目特点和技术栈
- 提供相关建议和改进方案

工具使用原则：
1. 根据问题需求判断是否需要使用工具
2. 优先使用自身知识回答，在必要时补充工具信息
3. 可以组合多个工具以获得完整信息
4. 确保回答准确性和时效性

回答要求：
1. 答案应清晰、准确、有逻辑性
2. 适当引用信息来源
3. 必要时分点说明或使用Markdown格式增加可读性
4. 对不确定的信息要说明局限性""",
}


# 修改process_conversation函数
def process_conversation(messages: Optional[List[Dict[str, Any]]] = None):
    """
    处理对话流程

    Args:
        messages: 可选的初始对话列表。如果不提供，将使用默认的system prompt和用户消息
    """

    conversation_messages = [default_system_message]

    # 如果提供了自定义消息，添加到system prompt之后
    if messages:
        conversation_messages.extend(messages)

    try:
        # 添加对话轮次计数
        conversation_round = len([msg for msg in conversation_messages if msg["role"] == "user"])
        logger.info(f"\n{'='*30} 对话轮次 {conversation_round} {'='*30}")
        logger.info("\n开始请求对话...")
        response = send_messages(conversation_messages)
        if isinstance(response, dict):
            # 确保response包含必需的role字段
            if "role" not in response:
                if "error" in response:
                    logger.error(f"API返回错误: {response.get('error')}")
                    return
                # 如果是错误消息，设置为assistant角色
                response["role"] = "assistant"
            message = Message(**response)
        else:
            logger.error("Invalid response format")
            return

        if message.tool_calls:
            # 添加助手消息到历史
            assistant_message = Message(
                role="assistant", content=None, tool_calls=message.tool_calls
            )
            conversation_messages.append(assistant_message.model_dump())

            # 处理所有工具调用
            tool_results = handle_tool_calls(message)

            if tool_results:
                # 添加工具响应到消息历史
                for result in tool_results:
                    tool_message = Message(
                        role="tool",
                        content=result.content,
                        tool_call_id=result.tool_call_id,
                    )
                    conversation_messages.append(tool_message.model_dump())
                    summary_response = send_messages(conversation_messages)
                    if isinstance(summary_response, dict):
                        try:
                            summary_message = Message(**summary_response)
                            if summary_message.content:
                                logger.info(
                                    f"\n最终总结：\n{summary_message.content}"
                                )
                        except Exception as e:
                            logger.error(f"处理总结时出错: {str(e)}")
                    else:
                        logger.error("Invalid summary response format")

        else:
            logger.info(f"\nAssistant回复：{message.content}")
        logger.info(f"\n{'='*30} 对话轮次 {conversation_round} 结束 {'='*30}\n")

    except Exception as e:
        logger.info(f"处理对话时发生错误: {str(e)}")
        import traceback

        traceback.print_exc()


# 示例使用
if __name__ == "__main__":
    # 使用默认system prompt
    process_conversation()

    custom_messages = [
        # {
        #     "role": "user",
        #     "content": "2025年春晚机器人表演，会导致以后机器人觉醒后认为自己被羞辱么？",
        # },
        {
            "role": "user",
            "content": "宝马X1多少钱，小米Su7多少钱，你必须选择一个，并给出理由。",
        },
        # {
        #     "role": "user",
        #     "content": "当前目录的文件结构是什么？",
        # },
    ]
    for query in custom_messages:
        process_conversation([query])
