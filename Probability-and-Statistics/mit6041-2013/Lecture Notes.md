### Lecture 1 Probability model and axioms

#### Concepts
1. Sample space, all possible outcomes. E.g., $\Omega = \{H, T\}$
	1. It needs to be complete and elements should be *mutual exclusive*. 
		1. Events *DO NOT* have to be mutual exclusive. Think the overlaps in Venn Diagram.
	2. It can be abstracted depending on experiment. E.g., $\{HHT\}$ if we care about order, $\{2H1T\}$ if we don't. 
	3. `(Tail, Head+Raining, Head+Not Raining)` is a valid sample space. As long as it is complete!
2. Events, E.g., $\{H\}, \{T\}, \varnothing$
	1. *Event* is a *subset*, rather than a point, in the sample space. 
	2. $\mathbf{P}(\mathit{s})$ is actually $\mathbf{P}(\mathit{\{s\}})$
3. Probability model
	1. Sample space
	2. Probability laws
4. Conditional probability
	1. It put the scope into a new universe. Probability laws in that universe is self-contained as non-conditional ones.
	2. "Cause-effect" and "inference": learn from new/partial evidence.

### Misc
1. *Set* vs *Probability*.
	1. Distinction is important coz their operations are different
	2. Additive Rule: $\mathbf{P}(A\cup B)=\mathbf{P}(A) + \mathbf{P}(B)$
	2. Multiplication Rule: $\mathbf{P}(A\cap B)=\mathbf{P}(A) * \mathbf{P}(A|B)$
2. *Good trick*: 
	1. To simplify question, try to make the problem discrete first.
		1. E.g., Romeo and Juliet (Example 1.5. p13) problem, instead of continuous one hour, simplify and assume that both can only arrive at every 15 minutes, then the sample space is discrete.
3. Fig 1.7 Bertrand's paradox. Different 'randomness' leads to different results.