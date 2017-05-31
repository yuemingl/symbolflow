package test.core;

import org.tensorflow.DataType;
import org.tensorflow.Graph;
import org.tensorflow.Output;

import test.operations.Add;
import test.operations.Mul;

public class MyTensor {
	protected String name;
	protected DataType dtype;
	protected Output output;
	
	public MyTensor(String name, DataType dtype) {
		this.name = name;
		this.dtype = dtype;
	}
	
	public MyTensor add(MyTensor other) {
		return new Add(this, other);
	}
	
	public MyTensor multiply(MyTensor other) {
		return new Mul(this, other);
	}
	
	public void buildGraph(Graph g) {
		this.output = g.opBuilder("Placeholder", name).setAttr("dtype", dtype).build().output(0);
	}
	
	public Output getOutput(int idx) {
		return this.output;
	}
	
	public DataType getDataType() {
		return this.dtype;
	}
	
	public String getName() {
		return this.name;
	}
}
