import java.util.ArrayList;

public class ParaTuning {

	public double tunePercepM(ArrayList<ArrayList<Integer>> trainingMatrix,
			ArrayList<ArrayList<Integer>> testMatrix, double[] etaCandiPercepM,
			int n) {
		System.out.println("n = " + n);
		double eta = 0;
		int[] testErrors = new int[etaCandiPercepM.length];
		// Iterate through each candidate
		for (int i = 0; i < etaCandiPercepM.length; i++) {
			PerceptronMargin percepM = new PerceptronMargin(n);
			eta = etaCandiPercepM[i];
			// Train classifier for 20 times on D1
			for (int times = 0; times < 20; times++) {
				for (int sample = 0; sample < trainingMatrix.size(); sample++) {
					ArrayList<Integer> data = trainingMatrix.get(sample);
					percepM.buildClassifier(data, eta);
				}
			}
			// Test performance on D2
			for (int test = 0; test < testMatrix.size(); test++) {
				ArrayList<Integer> data = testMatrix.get(test);
				int pred = percepM.classifyData(data);
				if (pred != data.get(data.size() - 1)) {
					testErrors[i] = testErrors[i] + 1;
				}
			}

			System.out.println("eta = "
					+ eta
					+ ", accu = "
					+ (1 - (double) testErrors[i]
							/ (double) trainingMatrix.size()));

		}

		System.out.println("----------------------------------------------");
		
		// return the optimal parameter value
		double[] result = getMinimum(testErrors);
		double para = etaCandiPercepM[(int) result[0]];
		return para;

	}

	public double tuneWinnow(ArrayList<ArrayList<Integer>> trainingMatrix,
			ArrayList<ArrayList<Integer>> testMatrix, double[] alphaWinnow,
			int n) {
		double alpha = 0;
		int[] testErrors = new int[alphaWinnow.length];
		// Iterate through each candidate
		for (int i = 0; i < alphaWinnow.length; i++) {
			Winnow winnow = new Winnow(n);
			alpha = alphaWinnow[i];
			// Train classifier for 20 times on D1
			for (int times = 0; times < 20; times++) {
				for (int sample = 0; sample < trainingMatrix.size(); sample++) {
					ArrayList<Integer> data = trainingMatrix.get(sample);
					winnow.buildClassifier(data, alpha, n);
				}
			}
			// Test performance on D2
			for (int test = 0; test < testMatrix.size(); test++) {
				ArrayList<Integer> data = testMatrix.get(test);
				int pred = winnow.classifyData(data);
				if (pred != data.get(testMatrix.get(0).size() - 1)) {
					testErrors[i] = testErrors[i] + 1;
				}
			}

			System.out.println("alpha = "
					+ alpha
					+ ", accu = "
					+ (1 - (double) testErrors[i]
							/ (double) trainingMatrix.size()));

		}

		System.out.println("----------------------------------------------");
		
		// return the optimal parameter value
		double[] result = getMinimum(testErrors);
		double para = alphaWinnow[(int) result[0]];
		return para;

	}

	public double[] tuneWinnowM(ArrayList<ArrayList<Integer>> trainingMatrix,
			ArrayList<ArrayList<Integer>> testMatrix, double[] alphaWinnowM,
			double[] gammaWinnowM, int n) {
		double alpha = 0;
		double gamma = 0;

		int[] testErrors = new int[gammaWinnowM.length * alphaWinnowM.length];
		// Iterate through each candidate
		for (int i = 0; i < alphaWinnowM.length; i++) {
			for (int j = 0; j < gammaWinnowM.length; j++) {
				WinnowMargin winnowM = new WinnowMargin(n);
				alpha = alphaWinnowM[i];
				gamma = gammaWinnowM[j];
				// Train classifier for 20 times on D1
				for (int times = 0; times < 20; times++) {
					for (int sample = 0; sample < trainingMatrix.size(); sample++) {
						ArrayList<Integer> data = trainingMatrix.get(sample);
						winnowM.buildClassifier(data, alpha, gamma, n);
					}
				}
				// Test performance on D2
				for (int test = 0; test < testMatrix.size(); test++) {
					ArrayList<Integer> data = testMatrix.get(test);
					int pred = winnowM.classifyData(data);
					if (pred != data.get(testMatrix.get(0).size() - 1)) {
						testErrors[i* gammaWinnowM.length + j] = testErrors[i* gammaWinnowM.length + j] + 1;
					}

				}

				System.out.println("alpha = "
						+ alpha
						+ ", gamma = "
						+ gamma
						+ ", accu = "
						+ (1 - (double) testErrors[i* gammaWinnowM.length + j]
								/ (double) trainingMatrix.size()));
			}

		}

		System.out.println("----------------------------------------------");
		
		// return the optimal parameter value
		double[] result = getMinimum(testErrors);
		int alphaIndex = (int) Math.floor(result[0]/gammaWinnowM.length); // index of alpha
		int gammaIndex = (int) result[0]%gammaWinnowM.length;// index of gamma
		double[] para = {alphaWinnowM[alphaIndex], gammaWinnowM[gammaIndex]};
		return para;

	}

	public double tuneAdaGrad(ArrayList<ArrayList<Integer>> trainingMatrix,
			ArrayList<ArrayList<Integer>> testMatrix, double[] etaCandiAdaGrad,
			int n) {

		double eta = 0;
		int[] testErrors = new int[etaCandiAdaGrad.length];
		// Iterate through each candidate
		for (int i = 0; i < etaCandiAdaGrad.length; i++) {
			AdaGrad adaGrad = new AdaGrad(n);
			eta = etaCandiAdaGrad[i];
			// Train classifier for 20 times on D1
			for (int times = 0; times < 20; times++) {
				for (int sample = 0; sample < trainingMatrix.size(); sample++) {
					// System.out.println("Training sample" + sample);
					ArrayList<Integer> data = trainingMatrix.get(sample);
					adaGrad.buildClassifier(data, eta);
				}
			}
			// Test performance on D2
			for (int test = 0; test < testMatrix.size(); test++) {
				ArrayList<Integer> data = testMatrix.get(test);
				int pred = adaGrad.classifyData(data);
				if (pred != data.get(data.size() - 1)) {
					testErrors[i] = testErrors[i] + 1;
				}
			}

			System.out.println("eta = "
					+ eta
					+ ", accu = "
					+ (1 - (double) testErrors[i]
							/ (double) trainingMatrix.size()));

		}
		System.out.println("----------------------------------------------");
		
		// return the optimal parameter value
		double[] result = getMinimum(testErrors);
		double para = etaCandiAdaGrad[(int) result[0]];
		return para;

	}

	public double[] getMinimum(int[] testErrors) {
		int smallest = 150000;
		double index = 500;
		for (int i = 0; i < testErrors.length; i++) {
			if (smallest > testErrors[i]) {
				smallest = testErrors[i];
				index = i;
			}
		}
		double[] result = { index, smallest };
		return result;
	}

}
