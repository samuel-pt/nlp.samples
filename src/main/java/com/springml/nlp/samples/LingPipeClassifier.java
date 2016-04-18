package com.springml.nlp.samples;

import java.io.File;

import com.aliasi.classify.Classification;
import com.aliasi.classify.Classified;
import com.aliasi.classify.DynamicLMClassifier;
import com.aliasi.classify.JointClassification;
import com.aliasi.classify.JointClassifier;
import com.aliasi.lm.NGramProcessLM;
import com.aliasi.util.AbstractExternalizable;
import com.aliasi.util.Files;

public class LingPipeClassifier {
    private static File TRAINING_DIR = new File(
            "/home/sam/work/projects/s/sample/nlp/nlp.samples/src/main/resources/fourNewsGroups/4news-train");

    private static File TESTING_DIR = new File(
            "/home/sam/work/projects/s/sample/nlp/nlp.samples/src/main/resources/fourNewsGroups/4news-test");

    private static String[] CATEGORIES = {
            "soc.religion.christian",
            "talk.religion.misc",
            "alt.atheism",
            "misc.forsale" };

    private JointClassifier<CharSequence> train() throws Exception {
        int nGramSize = 6;
        DynamicLMClassifier<NGramProcessLM> classifier = DynamicLMClassifier.
                createNGramProcess(CATEGORIES, nGramSize);

        for (String category : CATEGORIES) {
            File classDir = new File(TRAINING_DIR, category);
            String[] trainingFiles = classDir.list();
            for (String trainingFile : trainingFiles) {
                File file = new File(classDir, trainingFile);
                String trainingContent = Files.readFromFile(file, "UTF-8");

                Classification classification = new Classification(category);
                Classified<CharSequence> classified = new Classified<CharSequence>(trainingContent, classification);
                classifier.handle(classified);
            }
        }

        @SuppressWarnings("unchecked") // we created object so know it's safe
        JointClassifier<CharSequence> compiledClassifier = (JointClassifier<CharSequence>) AbstractExternalizable.compile(classifier);

        return compiledClassifier;
    }

    public static void main(String[] args) throws Exception {
        LingPipeClassifier classifier = new LingPipeClassifier();
        JointClassifier<CharSequence> jointClassifier = classifier.train();
        JointClassification classification = jointClassifier.classify("House Sale");
        System.out.println("classification " + classification.bestCategory());
        classification = jointClassifier.classify("Love");
        System.out.println("classification " + classification.bestCategory());
        classification = jointClassifier.classify("Religion");
        System.out.println("classification " + classification.bestCategory());
        classification = jointClassifier.classify("God helps");
        System.out.println("classification " + classification.bestCategory());

    }
}
