## How to tell a story
1. Tell stories with data; always quantify; Use 'I'.
1. Don't say "I don't know". Say "I can learn." and a quick example of previous learning experience
1. For L6 posistions:
    1. scope, scale, impact, complexity, ambiguity
    1. connect all dots when telling, i.e., the stakeholders, the customer etc.
1. She stressed on L6 vs L5: L5 can build the house when the requirements are clear; L6 only knows the budget, need to navigate the ambiguity, create a blue print, etc.

## Types of questions and how to answer
1. Type 1: knowledge check
    1. Example: explain the difference between L1 and L2 regularization
    1. definition, applicable situation, limitation, remedy
1. Type 2: backward looking
    1. Example: tell me about a previous project
    1. business statement (complexity), situation (scope/scale, # customer/data/user; team composistion) ~ 60s - 120s
    1. 'my' action/contribution + model + metric + result (improved by 10%), business impact ~ rest of the time
1. Type 3: forward looking
    1. Example: here is a situation; we'd like to build a solution...
    1. Ask clarifying questions; scope the problem before start answering.

## Particulars for this interview
1. Beginning: 2 min about myself.
1. End: 5-7 min for asking question.
1. science depth: strong/high impact project; what would I do differently; concept, limitation, what and why
1. coding (talk; trade-off; whiteboard; debug; time/space complexity)
1. ownership (act for customer, team, org, across orgs and long term; lesson learned document; my Azure compute instance through time)
    1. STAR

## Other best practices
1. take notes by hand
1. clarify
1. use AI to prepare--let it play the role of the interviewer.



## Stories
1. Account reconciliation
    1. shifting user; put an end to an open question: if LLM can follow the trend.
1. Period closing
    1. introduced LLM as user, adopted by the engineers.
1. Knowledge mining project, tech lead
1. VLM fine-tuning, tech lead
    1. The customer was a subbrand of a major fashion name, where hundreds of new items are introduced weekly. They were looking for a way to automate catelogue update, namely a model to automatically classify images of fashion product against a given taxonomy. 
    1. The stakeholders were the PM, the engineering team who owns the data platform and data, and the applied science team, who's responsible for training the model.
    1. The major challenges were: messy, incomplete taxonomy; limited data; tight timeline (need to make the ignite that year.)
    1. As the tech lead, I split the work in the following way: I was responsible for taxonomy creation and the end to end training pipeline; a more junior AS was working on public image collection, training pair generation and final evaluation. Later I also asked the manager to take over RAI due to limited time.
    1. The desired taxonomy need to meet to requirements: a) it should align with the model responsible for data generation (GPT-4o), e.g., does the model know this item as a 'sneaker' or a 'trainer'. b) it should have granularity in both classification and attributes. At the time we only have customer's online catelogue, which is a messy proxy of a taxonomy. Given the data generation depends on taxonomy and time is limited, my manager suggested me to use the customer's catelogue as the taxonomy, however I suspected that would downgrade the quality of the generated training date. I ended up working overtime and over the weekend to create the taxonomy. I managed to create a taxonomy consisted of 18 categories, such as footwear, accessory, each containing multiple classes (~100 in total) and subclasses. Moreover, I introduced level-specific attributes with inheritance and overriding rules. This made the taxnomomy much more token efficient and organized. 
    1. We leveraged Phi4-mm, 5.6B multimodal model as the base model for fine-tuning. Some highlights of the finetuning: a) I leveraged Azure Machine Learning Studio to do the training. Including dataset registration, training env creation, training/inferencing script creation and endpoint deployment. b) Conducted experiemnts on full SFT vs LoRA. 
    1. Results: We delivered a model with 90%+ top-1 accuracy on time. Not sure about the fate of the model after delivery to the engineering team because of change of project due to re-org. But this was one of the contributing models that helped our AS org had a spot in the new org.
    1. less populated natrually trained worse, even after downsampling. We did not have time to collect more data for those classes due to deadline.
    1. in retrospect, I would have done the project in sequential steps.
    
    1. Two events showing ownership: a) Volunteered to help the other AS for running part of the data generation jobs, otherwise we wouldn't be able to get close to 1Mil training data points; b) at a certain point, the other AS had arguments with the manager because of the pressure of timeline. I actively reached out to both and tried to get the stories from both sides.



1. https://www.amazon.jobs/en/jobs/10399493/senior-applied-scientist-fintelligence
1. Ownership
    1. Explanation: Leaders are owners. They think long term and don't sacrifice long-term value for short-term results. They act on behalf of the entire company, beyond just their own team. They never say "that's not my job."
    1. Example Questions
        1. Tell me about a time you took on a task that was outside of your core responsibilities.
            1. the Equinor 
            1. 
        1. Describe a situation where you identified a problem and took the initiative to fix it without being asked.
            1. the 80% image classification speedup
        1. Give me an example of when you made a decision that had a long-term positive impact even though it was costly in the short term.


1. tell me about yourself
1. biggest achievement
1. why do you want to work here
    1. the reputation of the company and the team


1. Ownership
    1. Explanation: Leaders are owners. They think long term and don't sacrifice long-term value for short-term results. They act on behalf of the entire company, beyond just their own team. They never say "that's not my job."
    1. Example Questions
        1. Tell me about a time you took on a task that was outside of your core responsibilities.
        1. Describe a situation where you identified a problem and took the initiative to fix it without being asked.
        1. Give me an example of when you made a decision that had a long-term positive impact even though it was costly in the short term.


1. Why did you leave MSFT
Toward the end of my tenure, Microsoft made a performance-related decision regarding my employment. That decision was difficult to reconcile with my previous performance reviews and contributions. In our final conversations, my manager acknowledged that Microsoft had changed, which aligns with my observation and general news, but I just did not expect it to happen in our org so soon.

1. backprop with numpy