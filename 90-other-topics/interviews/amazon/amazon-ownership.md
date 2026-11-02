### Indexing Pipeline — Ownership (Short STAR-B)
- S — Situation: Our pipeline processed tens of thousands of documents. Image I/O made indexing take more than a day, with roughly half of runs failing due to timeouts. The team had known about this bottleneck for a while.
- T — Task: Fixing the bottleneck wasn't assigned to me. Near the end of the project, after we trained an image-identification model, I saw an opportunity to solve it.
- A — Action: I independently designed and implemented a new pipeline: identify image bounding boxes, crop images directly from the original documents on the fly, and then classify them. This eliminated intermediate image storage.
- R — Result: Reduced end-to-end document parsing time by approximately 80%. Successful indexing went from taking more than a day, with frequent failures, to completing in a few hours, and the internal pipeline became much more stable.
- B — Business impact: Reduced wasted compute and engineering effort from failed runs. We provided the updated configuration to the customer near the end of the project, but didn't receive feedback confirming the downstream impact.

Ownership: Maintaining Customer Commitment Amid Organizational Uncertainty
Situation
- I was the tech lead on a customer-facing project, although formal ownership sat with my manager and PM.
- We had a customer meeting scheduled to present our latest results, but neither my manager nor the PM showed up, and neither had notified me beforehand.
- At the time, organizational priorities were shifting due to an impending reorganization. I later learned that both were occupied with the reorganization.
- The junior applied scientist on my team was also unsettled by the situation.
Task
- Although I wasn't the formal project owner, I felt responsible for honoring our commitment to the customer.
- I also wanted to make sure the junior scientist wasn't left to handle the unexpected situation alone and had an opportunity to present her work directly to the customer.
Action
- I decided to proceed with the meeting rather than cancel or postpone it because the usual project owners hadn't shown up.
- I brought the junior scientist along so we could present the latest results together, as originally planned.
- I wanted the customer to receive the update we'd committed to delivering, while also giving the junior scientist valuable experience presenting her work and engaging directly with the customer.
- I focused on keeping the customer engagement and technical update on track despite the internal uncertainty.
Result
- We delivered the planned technical update to the customer.
- The junior scientist had support during the meeting and an opportunity to showcase her work directly to the customer.
- I don't know whether this led to any further business impact, but I felt it was important to follow through on our commitment.
- The key lesson for me was that when organizational priorities shift and ownership becomes unclear, I shouldn't assume my responsibility ends with my formal role. I should use my judgment, follow through on commitments, and look for opportunities to support my teammates as well as our customers.