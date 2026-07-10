1. URL: https://www.mihaileric.com/The-Emperor-Has-No-Clothes/
2. How tools are introduced
	1. They listed in `SYSTEM_PROMPT`
	2. No `pydantic` schema anymore
3. The agent loop
	1. double `while` loop; notice the inner breaking condition--it always stops to get user input
	2. tool results are appended as 'user' content; I guess 'tool' would work too
