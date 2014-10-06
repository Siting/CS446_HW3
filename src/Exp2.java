import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Scanner;
import java.util.StringTokenizer;

public class Exp2 {

	public static void main(String[] args) throws IOException {

		int[] nCandi = { 40, 80, 120, 160, 200 };
		int n = 0;
		String fileNames = "";

		// Initialize error matrix
		ArrayList<ArrayList<Integer>> finalError = new ArrayList<ArrayList<Integer>>();

		// Iterate through each experiment
		for (int exp = 0; exp < nCandi.length; exp++) {

			ArrayList<Integer> errorExp = new ArrayList<Integer>();
			errorExp.add(nCandi[exp]);

			// get n & file name
			n = nCandi[exp];
			System.out.println("n = " + n + "-------------------------");
			if (exp == 0) {
				fileNames = "setN40.txt";
			} else if (exp == 1) {
				fileNames = "setN80.txt";
			} else if (exp == 2) {
				fileNames = "setN120.txt";
			} else if (exp == 3) {
				fileNames = "setN160.txt";
			} else if (exp == 4) {
				fileNames = "setN200.txt";
			}

			// Load file
			if (fileNames.isEmpty()) {
				System.out.println("need path");
				return;
			}
			File file = new File(fileNames);
			Scanner s = null;
			try {
				s = new Scanner(file);
			} catch (FileNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}

			// Save file to dataMatrix
			ArrayList<ArrayList<Integer>> dataMatrix = new ArrayList<ArrayList<Integer>>();
			while (s.hasNextLine()) {
				String line = s.nextLine();
				StringTokenizer tokenizer = new StringTokenizer(line, ",");
				ArrayList<Integer> dataRow = new ArrayList<Integer>();
				while (tokenizer.hasMoreElements()) {
					dataRow.add(Integer.valueOf(tokenizer.nextToken()));
				}
				dataMatrix.add(dataRow);
			}
			s.close();

			// Generate training & test data
			int numSamples = dataMatrix.size();
			// int numAttributes = dataMatrix.get(0).size();
			int numTenPercept = (int) Math.floor((double) 0.1 * numSamples);

			ArrayList<ArrayList<Integer>> trainingMatrix = new ArrayList<ArrayList<Integer>>();
			for (int i = 0; i < numTenPercept; i++) {
				trainingMatrix.add(dataMatrix.get(i));
			}

			ArrayList<ArrayList<Integer>> testMatrix = new ArrayList<ArrayList<Integer>>();
			for (int i = numTenPercept; i < 2 * numTenPercept; i++) {
				testMatrix.add(dataMatrix.get(i));
			}

			// Parameter candidate set
			double[] etaCandiPercepM = { 1.5, 0.25, 0.03, 0.005, 0.001 };
			double[] alphaCandiWinnow = { 1.1, 1.01, 1.005, 1.0005, 1.0001 };
			double[] alphaCandiWinnowM = { 1.1, 1.01, 1.005, 1.0005, 1.0001 };
			double[] gammaCandiWinnowM = { 2.0, 0.3, 0.04, 0.006, 0.001 };
			double[] etaCandiAdaGrad = { 1.5, 0.25, 0.03, 0.005, 0.001 };

			// Tuning
			double etaPercepM, alphaWinnow, alphaWinnowM, gammaWinnowM, etaAdaGrad;
			ParaTuning paraTune = new ParaTuning();
			etaPercepM = paraTune.tunePercepM(trainingMatrix, testMatrix,
					etaCandiPercepM, n);
			alphaWinnow = paraTune.tuneWinnow(trainingMatrix, testMatrix,
					alphaCandiWinnow, n);
			double[] result = paraTune.tuneWinnowM(trainingMatrix, testMatrix,
					alphaCandiWinnowM, gammaCandiWinnowM, n);
			alphaWinnowM = result[0];
			gammaWinnowM = result[1];
			etaAdaGrad = paraTune.tuneAdaGrad(trainingMatrix, testMatrix,
					etaCandiAdaGrad, n);
			System.out.println("Optimal values: etaPercepM = " + etaPercepM
					+ ", alphaWinnow = " + alphaWinnow + ", alphaWinnowM = "
					+ alphaWinnowM + ", gammaWinnowM = " + gammaWinnowM
					+ ", etaAdaGrad = " + etaAdaGrad);
			System.out
					.println("----------------------------------------------");

			// Training on the entire dataset
			Perceptron percep = new Perceptron(n);
			PerceptronMargin percepM = new PerceptronMargin(n);
			Winnow winnow = new Winnow(n);
			WinnowMargin winnowM = new WinnowMargin(n);
			AdaGrad adaGrad = new AdaGrad(n);

			for (int algo = 0; algo < 5; algo++) {
				int cumuE = 0;
				int correct = 0;
				for (int i = 0; i < dataMatrix.size(); i++) {

					ArrayList<Integer> data = dataMatrix.get(i);
					int pred = 0;

					if (algo == 0) {
						pred = percep.classifyData(data);
						percep.buildClassifier(data);
					} else if (algo == 1) {
						pred = percepM.classifyData(data);
						percepM.buildClassifier(data, etaPercepM);
					} else if (algo == 2) {
						pred = winnow.classifyData(data);
						winnow.buildClassifier(data, alphaWinnow, n);
					} else if (algo == 3) {
						pred = winnowM.classifyData(data);
						winnowM.buildClassifier(data, alphaWinnowM,
								gammaWinnowM, n);
					} else if (algo == 4) {
						pred = adaGrad.classifyData(data);
						adaGrad.buildClassifier(data, etaAdaGrad);
					}

					if (pred != data.get(data.size() - 1)) {
						cumuE = cumuE + 1;
						correct = 0;
					} else {
						correct++;
					}

					if (correct == 1000) {
						System.out.println("Algorithm converged");
						break;
					}

					if (i == dataMatrix.size() - 1) {
						i = 0;
					}

				}

				System.out.println("Algorithm " + algo + " has " + cumuE
						+ " mistakes");

				errorExp.add(cumuE);

			}

			finalError.add(errorExp);

		}

		// print to file
		PrintWriter out = new PrintWriter("exp2.txt");
		for (int i = 0; i < finalError.size(); i++) {
			for (int j = 0; j < finalError.get(0).size(); j++) {

				if (j == finalError.get(0).size() - 1) {
					out.println(finalError.get(i).get(j));
				} else {
					out.print(finalError.get(i).get(j) + ",");
				}
			}
		}
		out.close();

	}

}
