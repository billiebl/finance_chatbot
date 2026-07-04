import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def run_chat():
    # Points to your server file
    server_params = StdioServerParameters(command="python", args=["mcp_server.py"])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            print("--- FastMCP Chat (Type 'exit' to quit) ---")
            session_id = "user_123"

            while True:
                text = str(input("You: "))
                if text.lower() in ["exit", "quit"]: break

                # Call the 'chat' tool defined in the server
                response = await session.call_tool("chat", arguments={
                    "user_input": text,
                    "session_id": session_id
                })
                print(f"AI: {response.content[0].text}")


if __name__ == "__main__":
    asyncio.run(run_chat())