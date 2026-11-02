# Kebei’s Period closing story bank

Use this as a cue sheet, not a script. Lead with the section that matches the question; add detail only when asked.

## Project snapshot

- **Context:** Dynamics 365 Finance and Operations, an enterprise-grade ERP system with 5,000+ company users, for managing finace, manufacturing and supply chain. Period-closing schedule are the structured accounting steps taken at the end of a month or year to verify data accuracy, reconcile accounts, and finalize an organization's financial records to generate official reports. Example steps are post all bathces, reconcile accounts payable/receivable etc.
- **Goal:** Create agentic workflow for creating period-closing schedule with realistic constraints like due dates, calendars, worker assignments and dependencies.
- **My role:** leading applied scientist for creating the agentic skill and its evaluation
- **Team:** PM, platform engineer, applied science;

## Major challenges and resolution

- Challenge 1: No existing go-to documentation on period closing schedule generation. PM/SMEs knew how to use the platform but did not have the right terminology. (**ambiguity**)
- Resolution 1: I explored documentations and searched a lot of internal repos, and created a initial draft of the skill that the agent can follow to do a basic schedule geneartion, basing on previous period or template.
- Challenge 2: How to make the skill robust to realistic use cases? 
- Resolution 2: a) a lot of discussions with PMs for real-world use cases and failures, for e.g., wrong due dates because of leap year. b) create test cases accordingly. c) ask platform engineer to locate required operations/tables before adding to skill. For example, dependecies was not initially there. Many iterations of such interactions. (**leadership**?)
- Challenge 3: I was still not happy with the diversity of the test cases.
- Resolution 3: I reached out to the PM who owns the period closing platform and explored what was possible with the platform. From there I created calendars with holidays, assignable workers with similar names etc. (**ownership**)
- Challenge 4: The agentic harness lacked multi-turn capability, which was necessary for follow-up questions.
- Resolution 4: I hacked the source code of the harness and added another LLM to mimic user behavior. This enabled test cases where user made an inapplicable query and the agent had to reject and ask for correction. This made the test much more realistic.
- Challenge 5: once started running the test cases, the process was superslow, capped by the agentic harness.
- Resolution 5: a) Use python scripts to automate everything outside skill, e.g., seeding. b) Reached out to other Applied Scientist and learnt about sending test cases in parallel. The testing time was shortened by 80%--from two hours to <30 mins.

## Results
- an agentic skill with comphrehensive instructions on how to create period-closing schedule, including a lot of edge cases and blah blah..
- 30+ reusable, parameterized scenario templates covering diverse failure modes. These can be easily expanded to many more. Each test case was defined in a schema with multiple fields, some are
```python
{
  "seeding": <certain test cases require an existing closing schedule from previous period.>,
  "seedingVerfication": <check in the platform that the seeding is done accurately>,
  "userQuery": <user's main task>,
  "userAnswer": <user's answser to follow up questions>,
  "platformStateVerfication": <expected state of the platform after operation; e.g., task A is assigned to Alice>,
  "traceVerification": <expected key behavior. E.g., the agent assigned worker after checking with the user.>
}
```

## Impact
- This is a feature under development, so I do not know the adoption situation. But this will eventually be release to 5,000+ companies via the platform.
- The test cases was very robust, they were soon adapted to at least two other agentic clients/harnesses
- The introduction of 2nd LLM as user was adopted by the engineering team buidling skill evaluation repo, which will be leveraged by the whole AI ERP  organization with thousands of engineers.


1. ask ChatGPT to ask me questions
2. no Challenge 1
3. no explain why only 30+, leave that for follow up
3. no schema, no both status of the platform and trace inspection.
