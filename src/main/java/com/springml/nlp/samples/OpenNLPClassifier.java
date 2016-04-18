package com.springml.nlp.samples;

import java.io.File;
import java.io.IOException;

import opennlp.tools.doccat.DoccatModel;
import opennlp.tools.doccat.DocumentCategorizerME;
import opennlp.tools.doccat.DocumentSample;
import opennlp.tools.doccat.DocumentSampleStream;
import opennlp.tools.util.MarkableFileInputStreamFactory;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.PlainTextByLineStream;

public class OpenNLPClassifier {

    private DoccatModel train() {
        DoccatModel model = null;
        try {
            MarkableFileInputStreamFactory mfisFactory = new MarkableFileInputStreamFactory(
                    new File("/home/sam/work/projects/s/sample/nlp/nlp.samples/src/main/resources/en-animals.train"));
            ObjectStream<String> lineStream = new PlainTextByLineStream(mfisFactory, "UTF-8");
//            InputStream is = this.getClass().getResourceAsStream("en-animals.train");
//            ObjectStream<String> lineStream = new PlainTextByLineStream(is, "UTF-8");
            ObjectStream<DocumentSample> sampleStream = new DocumentSampleStream(lineStream);
            model = DocumentCategorizerME.train("en", sampleStream);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return model;
    }

    private void categorize(String text, DoccatModel model) {
        DocumentCategorizerME categorizer = new DocumentCategorizerME(model);
        double[] categorize = categorizer.categorize(text);
        String allResults = categorizer.getAllResults(categorize);
        String bestCategory = categorizer.getBestCategory(categorize);
        System.out.println("Categorize Results : " + categorize);
        System.out.println("All Results : " + allResults);
        System.out.println("Best Category : " + bestCategory);
    }

    public static void main(String[] args) {
        OpenNLPClassifier classifier = new OpenNLPClassifier();
        DoccatModel model = classifier.train();
        classifier.categorize("Because of its relatively low cost, chicken is one of the most used meats in the world. Nearly all parts of the bird can be used for food, and the meat can be cooked in many different ways", model);
        classifier.categorize("The felids are a rapidly evolving family of mammals that share a common ancestor only 10–15 million years ago[17] and include lions, tigers, cougars and many others",  model);
        classifier.categorize("Social animals are those animals which interact highly with other animals, usually of their own species (conspecifics), to the point of having a recognizable and distinct society", model);
        classifier.categorize("he Siberian Husky’s origins can be traced to the ancient Chukchi – an ancient tribe whose culture was based on the long-distance sled dog located near the Kolyma River Basin in Northern Siberia", model);
        classifier.categorize("The fossil record of snakes is relatively poor because snake skeletons are typically small and fragile making fossilization uncommon", model);
    }
}
