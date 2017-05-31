package symbolflow.operations;

import org.tensorflow.Graph;

import symbolflow.core.MyTensor;

public class Trans extends MyTensor {
	MyTensor t;

	public Trans(MyTensor t) {
		super(t.getName() + "_trans", t.getDataType());
		this.t = t;
	}

	public void buildGraph(Graph g) {
		t.buildGraph(g);
		this.output = g.opBuilder("Transpose", name).addInput(t.getOutput(0))
				.build().output(0);
	}
	
	public static MyTensor trans(MyTensor t) {
		return new Trans(t);
	}
}
