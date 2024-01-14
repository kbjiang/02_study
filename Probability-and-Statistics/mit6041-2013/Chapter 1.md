Date: 2023-12-30
Course:
Chapter: 1
Lecture: 1-4
Topic: #Probability #Probability/Foundation #Counting 

## Lectures
### Probability model
1. Sample space, all possible *outcomes*, also called elements. E.g., $\Omega = \{H, T\}$
	1. It needs to be complete and elements should be *mutual exclusive*. 
		1. Events *DO NOT* have to be mutual exclusive. Think the overlaps in Venn Diagram.
	2. It can be abstracted depending on experiment. E.g., $\{HHT\}$ if we care about order, $\{2H1T\}$ if we don't. 
	3. The outcome can be simple, but the process can be complicated. For example in the Monty Hall problem, one outcome is `win` and process could be `(prize behind door 1, initially pick door 2, switch)`. 
		1. In this case, sequential model can be used to visualize. See Misc below.
	4. `(Tail, Head+Raining, Head+Not Raining)` is a valid sample space. As long as it is complete!
2. Events, E.g., $\{H\}, \{T\}, \varnothing$
	1. *Event* is a *subset*, rather than a point, in the sample space. 
	2. $\mathbf{P}(\mathit{s})$ is actually $\mathbf{P}(\mathit{\{s\}})$
3. Probability model
	1. Sample space
	2. Probability laws
### Conditional Probability
1. Why deserve its own lecture?
	1. It's *sequential* in nature and nature if full of sequential events! See Example 1.11.
2. It put the scope into a *new universe*. Probability laws in that universe is self-contained as non-conditional ones.
	1. e.g., $\mathbf{P}(A\cup B|C)=\mathbf{P}(A|C)\mathbf{P}(B|C)$ given disjoint
3. *Cause-effect* and *Inference*
	1. *Cause-effect*: $\mathbf{P}(B|A_{i})$ where $A_i$ "causes" $B$. E.g., detection causes alarm.
	2. *Inference/Bayes' Rule*: given $B$, what's $\mathbf{P}(A_i|B)$. I.e., learn from new/partial evidence. E.g., given alarm, how likely detection.
4. This equation is used a lot.
	 1. $\mathbf{P}(A_1\cap A_2 \cdots \cap A_n ) = \mathbf{P}(A_1) \mathbf{P}(A_2 | A_1) \cdots \mathbf{P}(A_n | A_1 \cap A_2 \cdots \cap A_{n-1})$
	 2. This underlies the sequential model/diagram.
5. Interpret things like $\mathbf{P}(A|B\cap C)$. E.g.,  
	![[Pasted image 20231228102128.png]]
### Independence
1. Independence *does NOT mean no overlap (disjoint)*! Rather it's saying the knowledge of the event $B$ does not change the likelihood of event $A$.
	1. $\mathbf{P}(A|B)=\mathbf{P}(A)$
	2. Actually, disjoint events are super dependent. $A$ means definitely not $B$. 
		![[Pasted image 20240101090256.png|400]]
	3. Independence in $\Omega$ does not necessarily mean independence conditionally. See how $A$ and $B$ become totally dependent in universe $C$. 
		![[Pasted image 20240101091912.png|400]]
2. Pairwise independence is not enough. I.e., to say $A_1, A_2\text{ and } A_3$ are independent, must have $\textbf{P}(A_i\cap A_j)=\textbf{P}(A_i) \textbf{P}(A_j)$ and $\textbf{P}(A_1\cap A_2\cap A_3)=\textbf{P}(A_1) \textbf{P}(A_2) \textbf{P}(A_3)$ simultaneously.
	1. Example 1.22
### Counting
1. By stages
	1. Formally known as The counting principle
	2. *Do not mix!* Always be aware which stage we are counting. E.g., *First* combination of $n$ chess, *then* possible positions on the chessboard.
3. Permutation, Combination, partition
	1. Permutation: select $k$ element, *one by one (therefore order matters)*, from $n$ elements.
	2. Combination: select $k$-element *subset (therefore order does NOT matter)* from $n$ elements.
	3. Partition: 
		1. instead of $k$ out of $n$, think $k$ and $n-k$. Can be easily expended to $n_1, n_2,\cdots,n_r$
		2. E.g., 7 cards from 52-card deck, the num of combinations include exactly 3 aces is $\binom{4}{3} \times \binom{48}{4}$

## My comments
### Interesting/challenging
1. *Set* vs *Probability*.
	1. Distinction is important coz their operations are different
	2. Additive Rule: $\mathbf{P}(A\cup B)=\mathbf{P}(A) + \mathbf{P}(B)$
	2. Multiplication Rule: $\mathbf{P}(A\cap B)=\mathbf{P}(A) * \mathbf{P}(A|B)$
2. *Sequential* model for visualizing conditional/sequential probability. Fig 1.13
	1. E.g. 1.11. Instead of counting, do it sequentially! 
	2. This equation underlies the Sequential model. 
		1. $\mathbf{P}(A_1\cap A_2 \cdots \cap A_n ) = \mathbf{P}(A_1) \mathbf{P}(A_2 | A_1) \cdots \mathbf{P}(A_n | A_1 \cap A_2 \cdots \cap A_{n-1})$
	3. *Monty Hall*. 
		1. Intuition: coz the friend only picks from 2 and 3, so the sample space should be $\Omega = \{1, \{2, 3\}\}$ . Now it's easy to see that by switching, one ends up in the large subspace.
		2. Let $P_{i}$ denote the event where the prize is behind door $i$, $C_{i}$ denote the event where you initially choose door $i$, and $O_{i}$ denote the event where your friend opens door $i$.
			![[Pasted image 20231222161505.png|400]]
		3. Win when switching = Not selecting right door initially 
			![[Pasted image 20231222161742.png]]
3. Fig 1.7 Bertrand's paradox. Different 'randomness' leads to different results.
4. Example 1.24. 
	1. My initial thought is to calculate the probability of all three possible paths, $\mathbf{P}(ACEB), \mathbf{P}(ACFB)$ and $\mathbf{P}(ADB)$. However, since these paths are not independent, it does not make sense to take the product of them; neither are they disjoint so no addition. 
	2. The solution is to break this into subsystems consist of independent components. The `series` subsystem is straightforward, while the `parallel` subsystem contains independent failures.
		1. $\mathbf{P}(\text{series subsystem succeeds})=p_1 p_2 \cdots p_m$
		2. $\mathbf{P}(\text{parallel subsystem succeeds})=1 - (1-p_1) (1-p_2) \cdots (1-p_m)$
			![[Pasted image 20231228074342.png|600]]
5. Example 1.30 and 1.31
	1. $\sum\limits_{k=0}^{n} \binom{n}{k}=2^n.$ The sum of all numbers of $k$ element subsets $\equiv$ The number of all possible subsets.
6. *A problem I got wrong.* Total $n$ ball and $m$ are red. Now randomly select $k$, probability of having $i$ red.
	1. Correct way: partition by red and non-red.
	2. My way:
		1. Probability of $i$th ball is red $m/n$. This is true.
			1. E.g., $\mathbf{P}(2r)=\mathbf{P}(2r|1r)\mathbf{P}(1r) + \mathbf{P}(2r|\neg 1r)\mathbf{P}(\neg 1r)=\frac{m-1}{n-1}\frac{m}{n}+\frac{m}{n-1}(1-\frac{m}{n})=\frac{m}{n}$.
		2. Then it's "flip $k$ times and $i$ head" and easy to use binomial coefficient $\binom{k}{i} p^{i} (1-p)^{k-i}$ right? 
			1. *No!* because each flip in the sequence is not independent! The probability of this ball being red depends on the previous selections.
### Good trick 
1. To simplify question, try to make the problem discrete first.
		1. E.g., Romeo and Juliet (Example 1.5. p13) problem, instead of continuous one hour, simplify and assume that both can only arrive at every 15 minutes, then the sample space is discrete.