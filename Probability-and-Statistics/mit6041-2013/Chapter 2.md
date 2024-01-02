Date: 2024-01-02
Course:
Lecture: 5
Topic: #Probability #Probability/Foundation #Counting 

## Lectures
### Random variable, expectation and variance
1. Random variable
	1. the mapping/function
	2. no reasoning on expectation E[g(x)] != g(E[x])
2. Sample space, all possible *outcomes*, also called elements. E.g., $\Omega = \{H, T\}$
	1. It needs to be complete and elements should be *mutual exclusive*. 
		1. Events *DO NOT* have to be mutual exclusive. Think the overlaps in Venn Diagram.
	2. It can be abstracted depending on experiment. E.g., $\{HHT\}$ if we care about order, $\{2H1T\}$ if we don't. 
	3. The outcome can be simple, but the process can be complicated. For example in the Monty Hall problem, one outcome is `win` and process could be `(prize behind door 1, initially pick door 2, switch)`. 
		1. In this case, sequential model can be used to visualize. See Misc below.
	4. `(Tail, Head+Raining, Head+Not Raining)` is a valid sample space. As long as it is complete!
3. Events, E.g., $\{H\}, \{T\}, \varnothing$
	1. *Event* is a *subset*, rather than a point, in the sample space. 
	2. $\mathbf{P}(\mathit{s})$ is actually $\mathbf{P}(\mathit{\{s\}})$
4. Probability model
	1. Sample space
	2. Probability laws

## My comments