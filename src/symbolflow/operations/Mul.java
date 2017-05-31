package symbolflow.operations;

import org.tensorflow.Graph;

import symbolflow.core.MyTensor;

public class Mul extends MyTensor {
	MyTensor l;
	MyTensor r;

	public Mul(MyTensor l, MyTensor r) {
		super(l.getName() + "_mul_" + r.getName(), l.getDataType());
		this.l = l;
		this.r = r;
	}

	public void buildGraph(Graph g) {
		l.buildGraph(g);
		r.buildGraph(g);
		this.output = g.opBuilder("MatMul", name).addInput(l.getOutput(0))
				.addInput(r.getOutput(0)).build().output(0);
	}
}
