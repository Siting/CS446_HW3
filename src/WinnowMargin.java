import java.util.ArrayList;

/*
 * Changes from Perceptron: take in eta
 * 
 */

public class WinnowMargin extends Winnow {
	
	public WinnowMargin(int n) {
		super(n);
	}

	// Method: buildClassifier
	public void buildClassifier(ArrayList<Integer> data, double alpha,
			double gamma, int n) {

		int numAttributes = data.size() - 1; // also, the number of elements in
												// w
		int labelIndex = data.size() - 1;

		// compute wx+theta
		double computedY = 0;
		for (int i = 0; i < numAttributes; i++) {
			computedY += w[i] * data.get(i);
		}
		computedY = computedY + theta;

		// true or false
		int trueLabel = data.get(labelIndex);
		if (computedY * trueLabel <= gamma) {
			w = updateWeights(w, trueLabel, data, alpha);
		}

	}

}
