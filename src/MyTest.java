import org.tensorflow.DataType;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

public class MyTest {
	public static void main(String[] args) throws Exception {
		try (Graph g = new Graph()) {
			// Create a graph for Y = A * X
			transpose_A_times_X(g, new int[][] { { 1, 2 }, { 3, 4 } });

			// Execute the "Y = A * X" operation in a Session.
			try (Session s = new Session(g)) {
				Output feed = g.operation("X").output(0);
				Output fetch = g.operation("Y").output(0);
				try (Tensor x = Tensor.create(new int[][] { { 1 }, { 1 } })) {
					Tensor out = s.runner().feed(feed, x).fetch(fetch).run().get(0);
					int[][] rlt = new int[2][1];
					out.copyTo(rlt);
					printArray(rlt);
					System.out.println(out);
				}
			}
		}
	}

	public static Output constant(Graph g, String name, Object value) {
		try (Tensor t = Tensor.create(value)) {
			return g.opBuilder("Const", name).setAttr("dtype", t.dataType()).setAttr("value", t).build().output(0);
		}
	}

	public static Output placeholder(Graph g, String name, DataType dtype) {
		return g.opBuilder("Placeholder", name).setAttr("dtype", dtype).build().output(0);
	}

	public static Output addN(Graph g, Output... inputs) {
		return g.opBuilder("AddN", "AddN").addInputList(inputs).build().output(0);
	}

	public static Output matmul(Graph g, String name, Output a, Output b, boolean transposeA, boolean transposeB) {
		return g.opBuilder("MatMul", name).addInput(a).addInput(b).setAttr("transpose_a", transposeA)
				.setAttr("transpose_b", transposeB).build().output(0);
	}

	/**
	 * 
	 * @param g
	 * @param a
	 */
	public static void transpose_A_times_X(Graph g, int[][] a) {
		matmul(g, "Y", constant(g, "A", a), placeholder(g, "X", DataType.INT32), true, false);
	}

	public static void printArray(int[][] ary) {
		for (int i = 0; i < ary.length; i++) {
			for (int j = 0; j < ary[0].length; j++) {
				System.out.print(ary[i][j] + " ");
			}
			System.out.println();
		}
	}

}
