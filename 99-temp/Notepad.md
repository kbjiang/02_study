# 5-Day AI Agents Intensive Course with Google
### White paper
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