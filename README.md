# symbolflow

```java
public static void main(String[] args) throws Exception {
try (Graph g = new Graph()) {
	// Create a graph for Y = A * X + B
	MyTensor A = new MyConstTensor("A", new double[][] { { 1, 2 }, { 3, 4 } });
	MyTensor B = new MyConstTensor("B", new double[][] { { 10 }, { 100 } });
	MyTensor X = new MyTensor("X", DataType.DOUBLE);
	MyTensor Y = A * X + B;
	//MyTensor Y = trans(A) * X + B; //Exception in thread "main" java.lang.IllegalArgumentException: 1 inputs specified of 2 inputs in Op while building NodeDef 'A_trans' using Op<name=Transpose; signature=x:T, perm:Tperm -> y:T; attr=T:type; attr=Tperm:type,default=DT_INT32,allowed=[DT_INT32, DT_INT64]>
	Y.buildGraph(g);

	// Execute the "Y = A * X + B" operation in a Session.
	try (Session s = new Session(g)) {
		Output feedX = g.operation("X").output(0);
		Output fetch = g.operation(Y.getName()).output(0);
		try (Tensor t = Tensor.create(new double[][] { { 1 }, { 1 } })) {
			Tensor out = s.runner().feed(feedX, t).fetch(fetch).run().get(0);
			System.out.println(out);
			double[][] rlt = new double[2][1];
			out.copyTo(rlt);
			printArray(rlt);
		}
	}
}
}
```

```shell
2017-05-31 14:59:22.625017: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-05-31 14:59:22.625045: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2017-05-31 14:59:22.625051: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
2017-05-31 14:59:22.625055: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
DOUBLE tensor with shape [2, 1]
13.0 
107.0 
```
