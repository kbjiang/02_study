# 5-Day AI Agents Intensive Course with Google
### Day 1, White paper
1. The Vibe Coding to Agentic Engineering Spectrum
	1. The single biggest differentiator between the two ends is ==how outputs get verified.== In vibe coding, verification is optional; the developer runs the code and checks if it seems right. In agentic engineering, two mechanisms work together. ==Tests== verify the deterministic parts of the system: a function given this input produces that output. ==Evaluations==, or evals, verify the parts that are not deterministic: did the agent take the right trajectory of steps, choose the right tools, and produce a final response that meets the quality bar. Tests are checked by code; evals are checked by labelled datasets, scoring rubrics, and LM judges.  ![[Pasted image 20260615111157.png]]
2. Static and Dynamic context
	1. Static is always loaded: AGENTS.md, system instructions etc. They are expensive because they present in every interaction
	2. Dynamic is loaded on demand: RAG, tool results etc.
		1. skills... "The result is that an agent can carry dozens of specialized capabilities while paying the token cost for only the one it is actively using." ==How?==
	3. Their boundary is a first-class architectural decision 
	4. ![[Pasted image 20260615145313.png]]
3. "Success comes from giving agents success criteria rather than step-by-step instructions, then letting them iterate."
4. *Harness*: the surrounding machinery of an LLM
	1. Agent = Model (engine of a car) + Harness (belts, gears, assembly line) ![[Pasted image 20260615160033.png]]
	2. What is a Harness
		1. Instructions and Rule files, tools, ...
		2. It's the *team's surface area*, not the model provider's.
	3. Hooks and observability
5. Human's role: conductor vs orchestrator
6. The Economics of AI Development
	1. Capital Expenditure (CapEx)— the upfront investment to build something—and Operational Expenditure (OpEx)—the ongoing cost to run, fix, and maintain it.
### Day 2, White paper
1. MCP servers
	1. stdio: The host client launches the MCP server as a local background subprocess, passing JSON-RPC 2.0 messages over stdin and stdout.
	2. sse: 
	3. "Dynamically load tools from a registry only when needed, and ==drop them from context== when the task is complete to prevent attention dilution." How?
2. Agent-to-agent (A2A)
	1. Fragmentation: Every one of these specialist agents can be built by a different team, using different technologies.
	2. "This chaos of fragmentation is exactly what the Agent-to-Agent (A2A) protocol [A2A] is designed to standardize. A2A, originally developed by Google and now donated to the Linux Foundation, introduces a universal layer of communication for agentic systems. It acts as the lingua franca for the AI ecosystem, abstracting away networking transport nuances, the underlying frameworks, programming languages, and payload disparities."
	3. "It ensures that the central Orchestrator can discover, onboard, and collaborate with any specialist agent in the ecosystem, completely agnostic to how that specialist was built under the hood. Just as HTTP standardized the web, A2A standardizes the virtual workforce."
	4. "Without A2A, a developer might build an application and struggle to maintain its growing complexity. In the A2A era, that same developer can focus on perfecting a high-value niche— such as an agent that specializes in "Real-time Regulatory Compliance". Whether built as a sophisticated multi-agent monolith or a distributed app, these systems can now be exposed to the world as A2A-compliant agents. This means that a specialist vibe coded in one part of the world can be discovered and "hired" by another Orchestrator across the globe."
	5. "Ultimately, the A2A protocol transforms the act of building isolated agentic applications into building the foundational members of a global, interoperable digital workforce."
	6. "In a mature ecosystem, an application does not natively possess deep knowledge of every domain it touches. Instead, it acts as an Orchestrator—a central hub whose primary cognitive load is dedicated to understanding user intent, managing the overarching workflow, and delegating specific tasks to specialized, remote A2A agents."
### Day 3, Agent Skills
1. Skills enables one general-purpose agent to become a specialist across different things.
2. Skill vs multi-agent
	1. `single-agent-with-skills` much easier than `multiple-agent`
	2. Multi-agent remains the ==absolute right answer== when you have genuine parallelism, real capability boundaries (different access, different security postures, different external systems), hierarchical decomposition where the abstraction layers actually differ, adversarial or check-and-balance setups, sub-agent intercommunication, or heterogeneous models.
3. Agent only sees skill's *description* during routing
4. Evaluation
	1. ![[Pasted image 20260617114638.png]]
	2. ![[Pasted image 20260617114939.png]]
	3. Primary failure modes
		1. Trigger Failure
		2. Execution Failure
		3. Regression: adding the skill causes performance drops in existing library
		4. Token Budget Failure
	4. "When using LLM-as- Judge to score outputs at scale, remember two non-negotiables: swap the positions of the reference and actual outputs to eliminate ordering bias, and calibrate against human ratings until you hit 90% agreement."
	5. skills must graduate through strict tiers of authority: 
		1. Read-Only: LLM-as-Judge eval; 90% trigger accuracy. 
		2. Draft-Only (Human Review): Golden dataset of 20+ cases; human approval. 
		3. Action-Allowed: Full adversarial red-teaming; sustained success across multiple runs (not just a single lucky pass); no rollback events; sustained pass^k.