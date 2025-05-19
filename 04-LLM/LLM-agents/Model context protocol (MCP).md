## What is it
1. MCP acts as the middle layer between AI/user and external tools. 
	1. It's very similar to function calling, but provides a uniform protocol. 
		1. ![[Pasted image 20250505061818.png|800]]
		2. Since it's a protocol, each vendor such as slack or gmail, will need to follow it and provide the corresponding MCP configuration. Roughly speaking, think of MCP as the app store and each vendor develops its own app.
		3. Greatly *simplify* the use of tools! Now the user do NOT need to call APIs for each individual tools!
	2. A more detailed view
		1. ![[Pasted image 20250505062042.png|800]]
	3. Components
		1. MCP client: 
			1. think *cursor*, *vscode* or any IDE with MCP capability
			2. chooses which MCP server is appropriate for user input?
		2. MCP server
			1. one server specific to one tool
			2. provided by vendor of the tool
## How it works
1. ![[Pasted image 20250514162251.png|600]]
2. ![[Pasted image 20250514162850.png|600]]
3. local transport ![[Pasted image 20250514164804.png|600]]
4. The use of *mcp inspector* was good
	1. https://github.com/modelcontextprotocol/inspector
	2. to inspect the mcp server
5. [MCP Deep Dive - 深入研究 MCP 運作原理、架構與元件理解](https://youtu.be/6aOw26BVy4M)
	1. Good deep dive on components/mechanism; do not fully understand yet

## Hands-on
1. [MCP Explained in 15 Mins | Build Your Own Model Context Protocol Server Using Zapier & Cursor](https://youtu.be/SIwq53i1t4I)
	1. Visit zapier.com/mcp and add interested MCP and actions. E.g., gmail with "send_email" action.
	2. In cursor, copy the configuration from zapier and enable MCP.
2. [MCP是啥？技术原理是什么？一个视频搞懂MCP的一切。Windows系统配置MCP，Cursor,Cline 使用MCP](https://youtu.be/McNRkd5CxFY)