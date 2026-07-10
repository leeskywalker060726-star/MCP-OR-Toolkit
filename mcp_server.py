import json
import os
import sys
import asyncio
if os.name == 'nt' and sys.version_info >= (3, 8):
    mingw_bin_path = r"E:\mingw\mingw64\bin"
    if os.path.exists(mingw_bin_path):
        os.add_dll_directory(mingw_bin_path)

try:
    import exactor
except ImportError as e:
    print(f"\n[致命错误] 加载 exactor 失败: {e}", file=sys.stderr)
    print("如果是 'DLL load failed'，请确认上述 mingw_bin_path 是否正确，或者将该路径添加到系统的 PATH 环境变量中。", file=sys.stderr)
    sys.exit(1)


from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# ============================================================================
# ExactOR MCP Server (Data-Driven Architecture)
# 纯粹的调度器：配置与逻辑完全解耦。通过读取 JSONL 动态加载工具。
# ============================================================================

CONFIG_FILE = "tools_config.jsonl"
server = Server("ExactOR_Agent")

# 全局工具注册表与路由映射
registered_tools = []
tool_router = {}

def load_tools_config():
    """
    从 JSONL 配置文件动态加载可用算法及其 LLM Schema 描述
    """
    if not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError(f"[Error] 配置文件 {CONFIG_FILE} 不存在。")
    
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
                
            try:
                config = json.loads(line)
                name = config["name"]
                
                # 提取路由配置，若为 dynamic 则表示由 LLM 传参决定具体算法
                algo_type = config.pop("exact_algo_type", "dynamic")
                
                # 构造标准的 MCP Tool 对象
                tool = Tool(
                    name=name,
                    description=config.get("description", ""),
                    inputSchema=config.get("inputSchema", {})
                )
                
                registered_tools.append(tool)
                tool_router[name] = algo_type
                
            except json.JSONDecodeError as e:
                print(f"[Warning] 解析 {CONFIG_FILE} 第 {line_num} 行失败: {e}")

@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """
    将动态加载的工具列表暴露给大模型
    """
    return registered_tools

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
    """
    泛型化的调用入口，彻底消除硬编码的 if-else
    """
    try:
        if name not in tool_router:
            raise ValueError(f"未注册的工具调用: {name}")
            
        routing_algo = tool_router[name]
        
        # 1. 确定底层的算法类型 (静态路由 or 动态传参)
        algo_type = arguments.get("algo_type") if routing_algo == "dynamic" else routing_algo
        if not algo_type:
            raise ValueError(f"工具 {name} 缺失必要的 'algo_type' 参数。")
            
        # 2. 提取运筹学模型数据
        payload = arguments.get("payload", {})
        
        # 3. 跨语言调用 C++ 引擎
        result = exactor.solve(algo_type, payload)
        
        return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]
        
    except Exception as e:
        # 捕获 C++ 侧抛出的算理异常，安全返回给 LLM
        error_msg = f"ExactOR Backend Error: {str(e)}"
        return [TextContent(type="text", text=error_msg)]

async def main():
    print(">>> 正在加载 ExactOR 工具配置...")
    load_tools_config()
    print(f">>> 成功加载 {len(registered_tools)} 个算法工具。")
    print(">>> ExactOR MCP Server 正在启动 (Stdio 通信中) ...")
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="ExactOR_Agent",
                server_version="2.0.0", # 升级为 2.0 数据驱动版
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())