

## Chapter 13

### Decision theory = probability theory + utility theory

- Use __utility theory__ to represent and reason with preferences

### Product Rule

P( a ^ b ) = P( A | B )P( B )

![](imgs/5.png)

![](imgs/6.png)

# Chapter 14

![](imgs/7.png)

## Inference by Enumeration
Given, the following: 

P( B, E, A, J, M ) = P( J | A )P( M | A )P( A | E, B )P( B )P( E )

Find P( J | A )? 

$$=\sum_{B}^{\infty}\sum_{M}^{\infty}\sum_{E}^{\infty} \frac{P(b,e,j,m)}{P(a)}$$

## Evaluation Tree

**Don't fully understand**

![](imgs/8.png)
![](imgs/9.png)

# Chapter 16

![](imgs/1.png)
![](imgs/2.png)

## Decision Tree

![](imgs/3.png)

### Example 1


```python 
def DecisionTreeLearner(dataset):
    """[Figure 18.5]"""

    target, values = dataset.target, dataset.values

    def decision_tree_learning(examples, attrs, parent_examples=()):
        if len(examples) == 0:
            return plurality_value(parent_examples)
        elif all_same_class(examples):
            return DecisionLeaf(examples[0][target])
        elif len(attrs) == 0:
            return plurality_value(examples)
        else:
            A = choose_attribute(attrs, examples)
            tree = DecisionFork(A, dataset.attrnames[A], plurality_value(examples))
            for (v_k, exs) in split_by(A, examples):
                subtree = decision_tree_learning(
                    exs, removeall(A, attrs), examples)
                tree.add(v_k, subtree)
            return tree

    def plurality_value(examples):
        """Return the most popular target value for this set of examples.
        (If target is binary, this is the majority; otherwise plurality.)"""
        popular = argmax_random_tie(values[target],
                                    key=lambda v: count(target, v, examples))
        return DecisionLeaf(popular)

    def count(attr, val, examples):
        """Count the number of examples that have example[attr] = val."""
        return sum(e[attr] == val for e in examples)

    def all_same_class(examples):
        """Are all these examples in the same target class?"""
        class0 = examples[0][target]
        return all(e[target] == class0 for e in examples)

    def choose_attribute(attrs, examples):
        """Choose the attribute with the highest information gain."""
        return argmax_random_tie(attrs,
                                 key=lambda a: information_gain(a, examples))

    def information_gain(attr, examples):
        """Return the expected reduction in entropy from splitting by attr."""
        def I(examples):
            return information_content([count(target, v, examples)
                                        for v in values[target]])
        N = len(examples)
        remainder = sum((len(examples_i)/N) * I(examples_i)
                        for (v, examples_i) in split_by(attr, examples))
        return I(examples) - remainder

    def split_by(attr, examples):
        """Return a list of (val, examples) pairs for each val of attr."""
        return [(v, [e for e in examples if e[attr] == v])
                for v in values[attr]]

    return decision_tree_learning(dataset.examples, dataset.inputs)
```

# Chapter 18 

## Entropy 

    idk man

## Back-Propagation 

![](imgs/4.png)

```python
def BackPropagationLearner(dataset, net, learning_rate, epochs, activation=sigmoid):
    """[Figure 18.23] The back-propagation algorithm for multilayer networks"""
    # Initialise weights
    for layer in net:
        for node in layer:
            node.weights = random_weights(min_value=-0.5, max_value=0.5,
                                          num_weights=len(node.weights))

    examples = dataset.examples
    '''
    As of now dataset.target gives an int instead of list,
    Changing dataset class will have effect on all the learners.
    Will be taken care of later.
    '''
    o_nodes = net[-1]
    i_nodes = net[0]
    o_units = len(o_nodes)
    idx_t = dataset.target
    idx_i = dataset.inputs
    n_layers = len(net)

    inputs, targets = init_examples(examples, idx_i, idx_t, o_units)

for epoch in range(epochs):
    # Iterate over each example
    for e in range(len(examples)):
        i_val = inputs[e]
        t_val = targets[e]

        # Activate input layer
        for v, n in zip(i_val, i_nodes):
            n.value = v

        # Forward pass
        for layer in net[1:]:
            for node in layer:
                inc = [n.value for n in node.inputs]
                in_val = dotproduct(inc, node.weights)
                node.value = node.activation(in_val)

        # Initialize delta
        delta = [[] for _ in range(n_layers)]

        # Compute outer layer delta

        # Error for the MSE cost function
        err = [t_val[i] - o_nodes[i].value for i in range(o_units)]

        # The activation function used is relu or sigmoid function
        if node.activation == sigmoid:
            delta[-1] = [sigmoid_derivative(o_nodes[i].value) * err[i]
            for i in range(o_units)]
        elif node.activation == relu:
            delta[-1] = [relu_derivative(o_nodes[i].value) * err[i] 
            for i in range(o_units)]
        elif node.activation == tanh:
            delta[-1] = [tanh_derivative(o_nodes[i].value) * err[i] 
            for i in range(o_units)]
        elif node.activation == elu:
            delta[-1] = [elu_derivative(o_nodes[i].value) * err[i] 
            for i in range(o_units)]
        else:
            delta[-1] = [leaky_relu_derivative(o_nodes[i].value) * err[i] 
            for i in range(o_units)]


            # Backward pass
            h_layers = n_layers - 2
            for i in range(h_layers, 0, -1):
                layer = net[i]
                h_units = len(layer)
                nx_layer = net[i+1]

                # weights from each ith layer node to each i + 1th layer node
                w = [[node.weights[k] for node in nx_layer] for k in range(h_units)]

                if activation == sigmoid:
                    delta[i] = [sigmoid_derivative(layer[j].value) *
                    dotproduct(w[j], delta[i+1])
                            for j in range(h_units)]
                elif activation == relu:
                    delta[i] = [relu_derivative(layer[j].value) *
                    dotproduct(w[j], delta[i+1])
                            for j in range(h_units)]
                elif activation == tanh:
                    delta[i] = [tanh_derivative(layer[j].value) *
                    dotproduct(w[j], delta[i+1])
                            for j in range(h_units)]
                elif activation == elu:
                    delta[i] = [elu_derivative(layer[j].value) *
                    dotproduct(w[j], delta[i+1])
                            for j in range(h_units)]
                else:
                    delta[i] = [leaky_relu_derivative(layer[j].value) *
                    dotproduct(w[j], delta[i+1])
                            for j in range(h_units)]

            #  Update weights
            for i in range(1, n_layers):
                layer = net[i]
                inc = [node.value for node in net[i-1]]
                units = len(layer)
                for j in range(units):
                    layer[j].weights = vector_add(layer[j].weights,
                                                  scalar_vector_product(
                                                  learning_rate * delta[i][j], inc))

    return net
```

    
