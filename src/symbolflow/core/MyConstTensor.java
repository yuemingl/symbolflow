package symbolflow.core;

import java.lang.reflect.Array;

import org.tensorflow.DataType;
import org.tensorflow.Graph;
import org.tensorflow.Tensor;

public class MyConstTensor extends MyTensor {
	Object value;

	public MyConstTensor(String name, Object value) {
		super(name, dataTypeOf(value));
		this.value = value;
	}

	public void buildGraph(Graph g) {
		try (Tensor t = Tensor.create(value)) {
			this.output = g.opBuilder("Const", name).setAttr("dtype", dtype)
					.setAttr("value", t).build().output(0);
		}
	}

	private static DataType dataTypeOf(Object o) {
		if (o.getClass().isArray()) {
			if (Array.getLength(o) == 0) {
				throw new IllegalArgumentException(
						"cannot create Tensors with a 0 dimension");
			}
			// byte[] is a DataType.STRING scalar.
			Object e = Array.get(o, 0);
			if (Byte.class.isInstance(e) || byte.class.isInstance(e)) {
				return DataType.STRING;
			}
			return dataTypeOf(e);
		}
		if (Float.class.isInstance(o) || float.class.isInstance(o)) {
			return DataType.FLOAT;
		} else if (Double.class.isInstance(o) || double.class.isInstance(o)) {
			return DataType.DOUBLE;
		} else if (Integer.class.isInstance(o) || int.class.isInstance(o)) {
			return DataType.INT32;
		} else if (Long.class.isInstance(o) || long.class.isInstance(o)) {
			return DataType.INT64;
		} else if (Boolean.class.isInstance(o) || boolean.class.isInstance(o)) {
			return DataType.BOOL;
		} else {
			throw new IllegalArgumentException("cannot create Tensors of "
					+ o.getClass().getName());
		}
	}
}
