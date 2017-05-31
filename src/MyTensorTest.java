import org.tensorflow.DataType;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import test.core.MyConstTensor;
import test.core.MyTensor;
import static test.operations.Trans.*;

public class MyTensorTest {
	public static void main(String[] args) throws Exception {
		try (Graph g = new Graph()) {
			// Create a graph for Y = A * X + B
			MyTensor A = new MyConstTensor("A", new double[][] { { 1, 2 }, { 3, 4 } });
			MyTensor B = new MyConstTensor("B", new double[][] { { 10 }, { 100 } });
			MyTensor X = new MyTensor("X", DataType.DOUBLE);
			MyTensor Y = A * X + B;
			//MyTensor Y = trans(A) * X + B; //Exception in thread "main" java.lang.IllegalArgumentException: 1 inputs specified of 2 inputs in Op while building NodeDef 'A_trans' using Op<name=Transpose; signature=x:T, perm:Tperm -> y:T; attr=T:type; attr=Tperm:type,default=DT_INT32,allowed=[DT_INT32, DT_INT64]>
			Y.buildGraph(g);

			// Execute the "Y = A * X" operation in a Session.
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

	public static void printArray(double[][] ary) {
		for (int i = 0; i < ary.length; i++) {
			for (int j = 0; j < ary[0].length; j++) {
				System.out.print(ary[i][j] + " ");
			}
			System.out.println();
		}
	}
}
