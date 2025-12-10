import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.classification.BM25NBClassifier;
import org.apache.lucene.classification.ClassificationResult;
import org.apache.lucene.classification.Classifier;
import org.apache.lucene.classification.KNearestNeighborClassifier;
import org.apache.lucene.classification.SimpleNaiveBayesClassifier;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.classification.utils.ConfusionMatrixGenerator;
import org.apache.lucene.classification.utils.DatasetSplitter;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexableField;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.similarities.Similarity;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.ByteBuffersDirectory;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

/**
 * Clasificador Lucene para datos de Airbnb Los Angeles
 * 
 * Implementa dos tareas de clasificación usando 3 clasificadores:
 * - SimpleNaiveBayesClassifier
 * - KNearestNeighborClassifier (k=5)
 * - BM25NBClassifier
 * 
 * Tarea 1: Clasificación por neighbourhood_group_cleansed
 * Tarea 2: Clasificación por property_type (categorías macro)
 * 
 * COMPILACIÓN:
 * javac -cp "lucene-10.3.1/modules/*:lucene-10.3.1/modules-thirdparty/*:lib/*"
 * AirbnbClasificador.java
 * 
 * EJECUCIÓN:
 * java -cp ".:lucene-10.3.1/modules/*:lucene-10.3.1/modules-thirdparty/*:lib/*"
 * AirbnbClasificador --index-root ./index_root
 * 
 * ARGUMENTOS CLI:
 * --index-root <carpeta> : (OBLIGATORIO) Carpeta donde están los índices
 * --seed <n> : Semilla para división estratificada. Default: 1234
 * --k <n> : Número de vecinos para KNN. Default: 5
 */
public class AirbnbClasificador {

    private static final int DEFAULT_SEED = 1234;
    private static final int DEFAULT_K = 5;
    private static final double TRAIN_SPLIT = 0.7;
    private static final String FIELD_CONTENTS = "contents";
    private static final String FIELD_DESCRIPTION = "description";

    private final String indexRoot;
    private final int seed;
    private final int k;
    
    // Cache para contents reconstruido (evita recomputación)
    // Usa una clave basada en los campos del documento para mejor eficiencia
    private final Map<String, String> contentsCache = new HashMap<>();

    /**
     * Resultados de evaluación de un clasificador
     */
    public static class EvaluationResults {
        public final String classifierName;
        public final Map<String, Double> precisionByClass = new LinkedHashMap<>();
        public final Map<String, Double> recallByClass = new LinkedHashMap<>();
        public final Map<String, Double> f1ByClass = new LinkedHashMap<>();
        public double accuracy;
        public int truePositives;
        public int trueNegatives;
        public int falsePositives;
        public int falseNegatives;

        public EvaluationResults(String classifierName) {
            this.classifierName = classifierName;
        }
    }

    public AirbnbClasificador(String indexRoot, int seed, int k) {
        this.indexRoot = indexRoot;
        this.seed = seed;
        this.k = k;
    }

    public static void main(String[] args) {
        try {
            Config config = parseArgs(args);

            if (config.indexRoot == null) {
                System.err.println("Error: --index-root es obligatorio");
                System.err.println("Uso: java AirbnbClasificador --index-root <carpeta> [--seed <n>] [--k <n>]");
                System.exit(1);
            }

            AirbnbClasificador clasificador = new AirbnbClasificador(
                    config.indexRoot, config.seed, config.k);

            clasificador.ejecutar();

            System.exit(0);

        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }

    /**
     * Configuración CLI
     */
    public static class Config {
        String indexRoot;
        int seed = DEFAULT_SEED;
        int k = DEFAULT_K;
    }

    private static Config parseArgs(String[] args) {
        Config config = new Config();
        for (int i = 0; i < args.length; i++) {
            String arg = args[i];
            if (arg.startsWith("--")) {
                String value = (i + 1 < args.length && !args[i + 1].startsWith("--"))
                        ? args[++i]
                        : "";

                switch (arg) {
                    case "--index-root":
                        config.indexRoot = value;
                        break;
                    case "--seed":
                        try {
                            config.seed = value.isEmpty() ? DEFAULT_SEED : Integer.parseInt(value);
                        } catch (NumberFormatException e) {
                            config.seed = DEFAULT_SEED;
                        }
                        break;
                    case "--k":
                        try {
                            config.k = value.isEmpty() ? DEFAULT_K : Integer.parseInt(value);
                        } catch (NumberFormatException e) {
                            config.k = DEFAULT_K;
                        }
                        break;
                }
            }
        }
        return config;
    }

    /**
     * Ejecuta ambas tareas de clasificación
     */
    public void ejecutar() throws Exception {
        System.out.println("=== Clasificador Airbnb Lucene ===\n");
        System.out.println("Index root: " + indexRoot);
        System.out.println("Seed: " + seed);
        System.out.println("K (para KNN): " + k);
        System.out.println("Train/Test split: " + (TRAIN_SPLIT * 100) + "% / " + ((1 - TRAIN_SPLIT) * 100) + "%\n");

        Path indexPath = Paths.get(indexRoot, AirbnbIndexador.INDEX_PROPERTIES);
        if (!java.nio.file.Files.exists(indexPath)) {
            throw new IOException("Índice no encontrado: " + indexPath);
        }

        // Abrir índice
        Directory dir = FSDirectory.open(indexPath);
        IndexReader reader = DirectoryReader.open(dir);

        try {
            // Reutilizar analizador y similarity de AirbnbIndexador
            // Usa ClassicSimilarity para coincidir con BM25NBClassifier
            Analyzer analyzer = AirbnbIndexador.crearAnalizador();
            Similarity similarity = AirbnbIndexador.crearSimilarity();

            // Tarea 1: Clasificación por neighbourhood_group_cleansed
            System.out.println("=".repeat(80));
            System.out.println("TAREA 1: Clasificación por neighbourhood_group_cleansed");
            System.out.println("=".repeat(80));
            ejecutarTarea1(reader, analyzer, similarity);

            // Tarea 2: Clasificación por property_type
            System.out.println("\n" + "=".repeat(80));
            System.out.println("TAREA 2: Clasificación por property_type (categorías macro)");
            System.out.println("=".repeat(80));
            ejecutarTarea2(reader, analyzer, similarity);

            // Tarea 3: Clasificación por bedrooms (categorías: 0, 1, 2, 3, 4, 5+)
            System.out.println("\n" + "=".repeat(80));
            System.out.println("TAREA 3: Clasificación por bedrooms (0, 1, 2, 3, 4, 5+)");
            System.out.println("=".repeat(80));
            ejecutarTarea3(reader, analyzer, similarity);

        } finally {
            reader.close();
            dir.close();
        }
    }

    /**
     * Tarea 1: Clasificar por neighbourhood_group_cleansed
     * Clases: "city of los angeles", "other cities", "unincorporated areas"
     */
    private void ejecutarTarea1(IndexReader reader, Analyzer analyzer, Similarity similarity)
            throws Exception {
        String classField = "neighbourhood_group_cleansed";
        String textField = FIELD_CONTENTS;

        // Filtrar documentos que tienen contenido y clase válida
        List<Integer> validDocIds = obtenerDocumentosValidos(reader, textField, classField);
        if (validDocIds.isEmpty()) {
            System.err.println("No se encontraron documentos válidos para la tarea 1");
            System.err.println("Verifica que el campo 'neighbourhood_group_cleansed' esté indexado.");
            System.err.println("Ejecuta el indexador con --mode rebuild --force para reindexar.");
            return;
        }

        System.out.println("Documentos válidos: " + validDocIds.size());

        // Crear índice temporal con todos los documentos válidos
        Directory tempIndexDir = crearIndiceTemporalCompleto(reader, validDocIds, analyzer, similarity, classField, textField);
        
        // Usar DatasetSplitter de Lucene para dividir el dataset
        // Según la documentación: split(IndexReader, Directory, Directory, Directory, Analyzer, boolean, String, String...)
        IndexReader tempIndexReader = DirectoryReader.open(tempIndexDir);
        Directory trainDir = new ByteBuffersDirectory();
        Directory testDir = new ByteBuffersDirectory();
        Directory crossValidationDir = null; // No usamos cross-validation, solo train/test
        DatasetSplitter splitter = new DatasetSplitter(TRAIN_SPLIT, 1.0 - TRAIN_SPLIT);
        // split requiere: IndexReader originalIndex, Directory trainingIndex, Directory testIndex, 
        // Directory crossValidationIndex, Analyzer analyzer, boolean termVectors, String classFieldName, String... fieldNames
        splitter.split(tempIndexReader, trainDir, testDir, crossValidationDir, analyzer, false, classField, textField);
        tempIndexReader.close();
        
        IndexReader trainReader = DirectoryReader.open(trainDir);
        IndexReader testReader = DirectoryReader.open(testDir);
        
        System.out.println("Train: " + trainReader.numDocs() + " documentos");
        System.out.println("Test: " + testReader.numDocs() + " documentos");

        try {
            // Evaluar los 3 clasificadores
            List<EvaluationResults> results = new ArrayList<>();

            // 1. SimpleNaiveBayesClassifier
            System.out.println("\n--- SimpleNaiveBayesClassifier ---");
            SimpleNaiveBayesClassifier classifier1 = new SimpleNaiveBayesClassifier(trainReader, analyzer,
                    new MatchAllDocsQuery(), classField, textField);
            EvaluationResults r1 = evaluarClasificador(classifier1, testReader, classField, textField,
                    "SimpleNaiveBayesClassifier");
            results.add(r1);

            // 2. KNearestNeighborClassifier
            // Constructor: (IndexReader, Similarity, Analyzer, Query, int minDocFreq, int minTermFreq, int minWordLen, String classField, String... textFields)
            System.out.println("\n--- KNearestNeighborClassifier (k=" + k + ") ---");
            KNearestNeighborClassifier classifier2 = new KNearestNeighborClassifier(trainReader, similarity, analyzer,
                    new MatchAllDocsQuery(), 2, 5, 0, classField, textField);
            EvaluationResults r2 = evaluarClasificador(classifier2, testReader, classField, textField,
                    "KNearestNeighborClassifier");
            results.add(r2);

            // 3. BM25NBClassifier
            // NOTA: BM25NBClassifier crea su propio IndexSearcher internamente y usa la Similarity por defecto.
            // Usamos el clasificador tal como está diseñado, sin workarounds.
            System.out.println("\n--- BM25NBClassifier ---");
            try {
                BM25NBClassifier classifier3 = new BM25NBClassifier(trainReader, analyzer, new MatchAllDocsQuery(),
                        classField, textField);
                EvaluationResults r3 = evaluarClasificador(classifier3, testReader, classField, textField,
                        "BM25NBClassifier");
                results.add(r3);
            } catch (Exception e) {
                System.err.println("Error creando BM25NBClassifier: " + e.getMessage());
                e.printStackTrace();
            }

            // Mostrar tabla comparativa
            mostrarTablaComparativa(results, "Tarea 1: neighbourhood_group_cleansed");

        } finally {
            trainReader.close();
            testReader.close();
        }
    }

    /**
     * Discretiza el número de bedrooms en categorías: "0", "1", "2", "3", "4", "5+"
     */
    public static String discretizarBedrooms(Integer bedrooms) {
        if (bedrooms == null) {
            return "0";
        }
        if (bedrooms == 0) {
            return "0";
        } else if (bedrooms == 1) {
            return "1";
        } else if (bedrooms == 2) {
            return "2";
        } else if (bedrooms == 3) {
            return "3";
        } else if (bedrooms == 4) {
            return "4";
        } else {
            return "5+";
        }
    }

    /**
     * Tarea 2: Clasificar por property_type (categorías macro)
     * Usa classifyPropertyType() para obtener la categoría
     */
    private void ejecutarTarea2(IndexReader reader, Analyzer analyzer, Similarity similarity)
            throws Exception {
        String classField = "property_type_category"; // Campo virtual que crearemos
        String textField = FIELD_CONTENTS;

        // Crear un índice temporal con property_type_category
        // O mejor: procesar documentos y crear las clases dinámicamente
        // Necesitamos mapear property_type -> categoría usando classifyPropertyType

        // Filtrar documentos válidos y crear campo virtual property_type_category
        List<DocumentWithClass> validDocs = obtenerDocumentosConCategoria(reader, textField);
        if (validDocs.isEmpty()) {
            System.err.println("No se encontraron documentos válidos para la tarea 2");
            return;
        }

        System.out.println("Documentos válidos: " + validDocs.size());

        // Crear índice temporal con todos los documentos válidos
        Directory tempIndexDir = crearIndiceTemporalCompletoConCategoria(validDocs, analyzer, similarity, classField);
        
        // Usar DatasetSplitter de Lucene para dividir el dataset
        // Según la documentación: split(IndexReader, Directory, Directory, Directory, Analyzer, boolean, String, String...)
        IndexReader tempIndexReader = DirectoryReader.open(tempIndexDir);
        Directory trainDir = new ByteBuffersDirectory();
        Directory testDir = new ByteBuffersDirectory();
        Directory crossValidationDir = null; // No usamos cross-validation, solo train/test
        DatasetSplitter splitter = new DatasetSplitter(TRAIN_SPLIT, 1.0 - TRAIN_SPLIT);
        // split requiere: IndexReader originalIndex, Directory trainingIndex, Directory testIndex, 
        // Directory crossValidationIndex, Analyzer analyzer, boolean termVectors, String classFieldName, String... fieldNames
        splitter.split(tempIndexReader, trainDir, testDir, crossValidationDir, analyzer, false, classField, textField);
        tempIndexReader.close();
        
        IndexReader trainReader = DirectoryReader.open(trainDir);
        IndexReader testReader = DirectoryReader.open(testDir);
        
        System.out.println("Train: " + trainReader.numDocs() + " documentos");
        System.out.println("Test: " + testReader.numDocs() + " documentos");

        try {
            // Evaluar los 3 clasificadores
            List<EvaluationResults> results = new ArrayList<>();

            // 1. SimpleNaiveBayesClassifier
            System.out.println("\n--- SimpleNaiveBayesClassifier ---");
            SimpleNaiveBayesClassifier classifier1 = new SimpleNaiveBayesClassifier(trainReader, analyzer,
                    new MatchAllDocsQuery(), classField, textField);
            EvaluationResults r1 = evaluarClasificador(classifier1, testReader, classField, textField,
                    "SimpleNaiveBayesClassifier");
            results.add(r1);

            // 2. KNearestNeighborClassifier
            // Constructor: (IndexReader, Similarity, Analyzer, Query, int minDocFreq, int minTermFreq, int minWordLen, String classField, String... textFields)
            System.out.println("\n--- KNearestNeighborClassifier (k=" + k + ") ---");
            KNearestNeighborClassifier classifier2 = new KNearestNeighborClassifier(trainReader, similarity, analyzer,
                    new MatchAllDocsQuery(), 2, 5, 0, classField, textField);
            EvaluationResults r2 = evaluarClasificador(classifier2, testReader, classField, textField,
                    "KNearestNeighborClassifier");
            results.add(r2);

            // 3. BM25NBClassifier
            // NOTA: BM25NBClassifier crea su propio IndexSearcher internamente y usa la Similarity por defecto.
            // Usamos el clasificador tal como está diseñado, sin workarounds.
            System.out.println("\n--- BM25NBClassifier ---");
            try {
                BM25NBClassifier classifier3 = new BM25NBClassifier(trainReader, analyzer, new MatchAllDocsQuery(),
                        classField, textField);
                EvaluationResults r3 = evaluarClasificador(classifier3, testReader, classField, textField,
                        "BM25NBClassifier");
                results.add(r3);
            } catch (Exception e) {
                System.err.println("Error creando BM25NBClassifier: " + e.getMessage());
                e.printStackTrace();
            }

            // Mostrar tabla comparativa
            mostrarTablaComparativa(results, "Tarea 2: property_type (categorías macro)");

        } finally {
            trainReader.close();
            testReader.close();
        }
    }

    /**
     * Tarea 3: Clasificar por bedrooms (categorías: 0, 1, 2, 3, 4, 5+)
     * Discretiza el campo numérico bedrooms y clasifica basándose en la descripción
     */
    private void ejecutarTarea3(IndexReader reader, Analyzer analyzer, Similarity similarity)
            throws Exception {
        String classField = "bedrooms_category"; // Campo virtual
        String textField = FIELD_CONTENTS;

        // Obtener documentos con su categoría de bedrooms
        List<DocumentWithClass> validDocs = obtenerDocumentosConCategoriaBedrooms(reader, textField);
        if (validDocs.isEmpty()) {
            System.err.println("No se encontraron documentos válidos para la tarea 3");
            return;
        }

        System.out.println("Documentos válidos: " + validDocs.size());

        // Crear índice temporal con todos los documentos válidos
        Directory tempIndexDir = crearIndiceTemporalCompletoConCategoria(validDocs, analyzer, similarity, classField);
        
        // Usar DatasetSplitter de Lucene para dividir el dataset
        // Según la documentación: split(IndexReader, Directory, Directory, Directory, Analyzer, boolean, String, String...)
        IndexReader tempIndexReader = DirectoryReader.open(tempIndexDir);
        Directory trainDir = new ByteBuffersDirectory();
        Directory testDir = new ByteBuffersDirectory();
        Directory crossValidationDir = null; // No usamos cross-validation, solo train/test
        DatasetSplitter splitter = new DatasetSplitter(TRAIN_SPLIT, 1.0 - TRAIN_SPLIT);
        // split requiere: IndexReader originalIndex, Directory trainingIndex, Directory testIndex, 
        // Directory crossValidationIndex, Analyzer analyzer, boolean termVectors, String classFieldName, String... fieldNames
        splitter.split(tempIndexReader, trainDir, testDir, crossValidationDir, analyzer, false, classField, textField);
        tempIndexReader.close();
        
        IndexReader trainReader = DirectoryReader.open(trainDir);
        IndexReader testReader = DirectoryReader.open(testDir);
        
        System.out.println("Train: " + trainReader.numDocs() + " documentos");
        System.out.println("Test: " + testReader.numDocs() + " documentos");

        try {
            // Evaluar los 3 clasificadores
            List<EvaluationResults> results = new ArrayList<>();

            // 1. SimpleNaiveBayesClassifier
            System.out.println("\n--- SimpleNaiveBayesClassifier ---");
            SimpleNaiveBayesClassifier classifier1 = new SimpleNaiveBayesClassifier(trainReader, analyzer,
                    new MatchAllDocsQuery(), classField, textField);
            EvaluationResults r1 = evaluarClasificador(classifier1, testReader, classField, textField,
                    "SimpleNaiveBayesClassifier");
            results.add(r1);

            // 2. KNearestNeighborClassifier
            // Constructor: (IndexReader, Similarity, Analyzer, Query, int minDocFreq, int minTermFreq, int minWordLen, String classField, String... textFields)
            System.out.println("\n--- KNearestNeighborClassifier (k=" + k + ") ---");
            KNearestNeighborClassifier classifier2 = new KNearestNeighborClassifier(trainReader, similarity, analyzer,
                    new MatchAllDocsQuery(), 2, 5, 0, classField, textField);
            EvaluationResults r2 = evaluarClasificador(classifier2, testReader, classField, textField,
                    "KNearestNeighborClassifier");
            results.add(r2);

            // 3. BM25NBClassifier
            // NOTA: BM25NBClassifier crea su propio IndexSearcher internamente y usa la Similarity por defecto.
            // Usamos el clasificador tal como está diseñado, sin workarounds.
            System.out.println("\n--- BM25NBClassifier ---");
            try {
                BM25NBClassifier classifier3 = new BM25NBClassifier(trainReader, analyzer, new MatchAllDocsQuery(),
                        classField, textField);
                EvaluationResults r3 = evaluarClasificador(classifier3, testReader, classField, textField,
                        "BM25NBClassifier");
                results.add(r3);
            } catch (Exception e) {
                System.err.println("Error creando BM25NBClassifier: " + e.getMessage());
                e.printStackTrace();
            }

            // Mostrar tabla comparativa
            mostrarTablaComparativa(results, "Tarea 3: bedrooms (0, 1, 2, 3, 4, 5+)");

        } finally {
            trainReader.close();
            testReader.close();
        }
    }

    /**
     * Obtiene documentos válidos (con contenido y clase)
     */
    private List<Integer> obtenerDocumentosValidos(IndexReader reader, String textField, String classField)
            throws IOException {
        List<Integer> validIds = new ArrayList<>();
        IndexSearcher searcher = new IndexSearcher(reader);
        TopDocs topDocs = searcher.search(new MatchAllDocsQuery(), reader.numDocs());

        for (ScoreDoc scoreDoc : topDocs.scoreDocs) {
            Document doc = searcher.storedFields().document(scoreDoc.doc);
            
            // Obtener texto del campo contents o reconstruirlo
            String text = doc.get(textField);
            if (text == null || text.trim().isEmpty()) {
                text = reconstruirContents(doc);
            }
            
            String classValue = doc.get(classField);

            if (text != null && !text.trim().isEmpty() && classValue != null && !classValue.trim().isEmpty()) {
                validIds.add(scoreDoc.doc);
            }
        }

        return validIds;
    }

    /**
     * Obtiene documentos con su categoría de bedrooms (discretizado)
     */
    private List<DocumentWithClass> obtenerDocumentosConCategoriaBedrooms(IndexReader reader, String textField)
            throws IOException {
        List<DocumentWithClass> validDocs = new ArrayList<>();
        IndexSearcher searcher = new IndexSearcher(reader);
        TopDocs topDocs = searcher.search(new MatchAllDocsQuery(), reader.numDocs());

        for (ScoreDoc scoreDoc : topDocs.scoreDocs) {
            Document doc = searcher.storedFields().document(scoreDoc.doc);
            
            // Obtener texto del campo contents o reconstruirlo
            String text = doc.get(textField);
            if (text == null || text.trim().isEmpty()) {
                text = reconstruirContents(doc);
            }
            
            // Obtener bedrooms (campo numérico)
            String bedroomsStr = doc.get("bedrooms");
            Integer bedrooms = null;
            if (bedroomsStr != null && !bedroomsStr.trim().isEmpty()) {
                try {
                    bedrooms = Integer.parseInt(bedroomsStr.trim());
                } catch (NumberFormatException e) {
                    // Ignorar si no se puede parsear
                }
            }

            if (text != null && !text.trim().isEmpty() && bedrooms != null) {
                // Discretizar bedrooms en categoría
                String category = discretizarBedrooms(bedrooms);
                validDocs.add(new DocumentWithClass(doc, category));
            }
        }

        return validDocs;
    }

    /**
     * Obtiene documentos con su categoría de property_type
     */
    private List<DocumentWithClass> obtenerDocumentosConCategoria(IndexReader reader, String textField)
            throws IOException {
        List<DocumentWithClass> validDocs = new ArrayList<>();
        IndexSearcher searcher = new IndexSearcher(reader);
        TopDocs topDocs = searcher.search(new MatchAllDocsQuery(), reader.numDocs());

        for (ScoreDoc scoreDoc : topDocs.scoreDocs) {
            Document doc = searcher.storedFields().document(scoreDoc.doc);
            
            // Obtener texto del campo contents o reconstruirlo
            String text = doc.get(textField);
            if (text == null || text.trim().isEmpty()) {
                text = reconstruirContents(doc);
            }
            
            // Intentar obtener property_type_original, si no existe usar property_type
            String propertyType = doc.get("property_type_original");
            if (propertyType == null || propertyType.trim().isEmpty()) {
                propertyType = doc.get("property_type");
            }

            if (text != null && !text.trim().isEmpty() && propertyType != null && !propertyType.trim().isEmpty()) {
                // Clasificar property_type en categoría macro
                String category = AirbnbIndexador.classifyPropertyType(propertyType).toLowerCase();
                validDocs.add(new DocumentWithClass(doc, category));
            }
        }

        return validDocs;
    }

    /**
     * Wrapper para documento con su clase
     */
    private static class DocumentWithClass {
        final Document doc;
        final String category;

        DocumentWithClass(Document doc, String category) {
            this.doc = doc;
            this.category = category;
        }
    }

    /**
     * Divide dataset estratificado por clase
     * @deprecated Usar DatasetSplitter de Lucene en su lugar
     */
    @Deprecated
    private DatasetSplit dividirDatasetEstratificado(IndexReader reader, List<Integer> docIds, String classField,
            int seed) throws IOException {
        // Agrupar por clase
        Map<String, List<Integer>> docsByClass = new HashMap<>();
        IndexSearcher searcher = new IndexSearcher(reader);
        
        for (int docId : docIds) {
            Document doc = searcher.storedFields().document(docId);
            String classValue = doc.get(classField);
            if (classValue != null) {
                String normalizedClass = classValue.toLowerCase().trim();
                docsByClass.computeIfAbsent(normalizedClass, k -> new ArrayList<>()).add(docId);
            }
        }

        // Dividir cada clase estratificadamente
        Random random = new Random(seed);
        List<Integer> trainIds = new ArrayList<>();
        List<Integer> testIds = new ArrayList<>();

        for (Map.Entry<String, List<Integer>> entry : docsByClass.entrySet()) {
            List<Integer> classDocs = entry.getValue();
            Collections.shuffle(classDocs, random);

            int trainSize = (int) (classDocs.size() * TRAIN_SPLIT);
            trainIds.addAll(classDocs.subList(0, trainSize));
            testIds.addAll(classDocs.subList(trainSize, classDocs.size()));
        }

        return new DatasetSplit(trainIds, testIds);
    }

    /**
     * Divide dataset estratificado con categorías
     * @deprecated Usar DatasetSplitter de Lucene en su lugar
     */
    @Deprecated
    private DatasetSplitDocs dividirDatasetEstratificadoConCategoria(List<DocumentWithClass> docs, int seed) {
        // Agrupar por categoría
        Map<String, List<DocumentWithClass>> docsByClass = new HashMap<>();
        for (DocumentWithClass doc : docs) {
            docsByClass.computeIfAbsent(doc.category, k -> new ArrayList<>()).add(doc);
        }

        // Dividir cada clase estratificadamente
        Random random = new Random(seed);
        List<DocumentWithClass> trainDocs = new ArrayList<>();
        List<DocumentWithClass> testDocs = new ArrayList<>();

        for (Map.Entry<String, List<DocumentWithClass>> entry : docsByClass.entrySet()) {
            List<DocumentWithClass> classDocs = entry.getValue();
            Collections.shuffle(classDocs, random);

            int trainSize = (int) (classDocs.size() * TRAIN_SPLIT);
            trainDocs.addAll(classDocs.subList(0, trainSize));
            testDocs.addAll(classDocs.subList(trainSize, classDocs.size()));
        }

        return new DatasetSplitDocs(trainDocs, testDocs);
    }

    /**
     * Wrapper para división de dataset
     */
    private static class DatasetSplit {
        final List<Integer> trainDocIds;
        final List<Integer> testDocIds;
        final List<DocumentWithClass> trainDocs;
        final List<DocumentWithClass> testDocs;

        DatasetSplit(List<Integer> trainDocIds, List<Integer> testDocIds) {
            this.trainDocIds = trainDocIds;
            this.testDocIds = testDocIds;
            this.trainDocs = null;
            this.testDocs = null;
        }

    }
    
    /**
     * Wrapper para división de dataset con documentos
     */
    private static class DatasetSplitDocs {
        final List<DocumentWithClass> trainDocs;
        final List<DocumentWithClass> testDocs;

        DatasetSplitDocs(List<DocumentWithClass> trainDocs, List<DocumentWithClass> testDocs) {
            this.trainDocs = trainDocs;
            this.testDocs = testDocs;
        }
    }

    /**
     * Genera una clave única para el cache basada en los campos del documento
     */
    private String generarCacheKey(Document doc) {
        // Usar campos que identifican únicamente el documento
        StringBuilder key = new StringBuilder();
        String name = doc.get("name");
        String id = doc.get("id");
        if (name != null) key.append(name);
        if (id != null) key.append("|").append(id);
        // Si no hay name ni id, usar todos los campos relevantes
        if (key.length() == 0) {
            String desc = doc.get("description");
            if (desc != null && desc.length() > 0) {
                key.append(desc.substring(0, Math.min(50, desc.length())));
            }
        }
        return key.toString();
    }
    
    /**
     * Reconstruye el campo contents desde los campos stored disponibles
     * (igual que en AirbnbIndexador.crearDocumentoPropiedad)
     * Usa cache para evitar recomputación
     */
    private String reconstruirContents(Document doc) {
        // Verificar cache primero usando clave basada en campos del documento
        String cacheKey = generarCacheKey(doc);
        String cached = contentsCache.get(cacheKey);
        if (cached != null) {
            return cached;
        }
        
        StringBuilder contents = new StringBuilder();
        
        // 1. Name
        String name = doc.get("name");
        if (name != null) contents.append(name).append(" ");
        
        // 2. Description
        String description = doc.get("description");
        if (description != null) contents.append(description).append(" ");
        
        // 3. Neighborhood Overview
        String neighborhoodOverview = doc.get("neighborhood_overview");
        if (neighborhoodOverview != null) contents.append(neighborhoodOverview).append(" ");
        
        // 4. Neighbourhood Cleansed
        String neighbourhood = doc.get("neighbourhood_cleansed_original");
        if (neighbourhood == null) neighbourhood = doc.get("neighbourhood_cleansed");
        if (neighbourhood != null) contents.append(neighbourhood).append(" ");
        
        // 5. Property Type
        String propertyType = doc.get("property_type_original");
        if (propertyType == null) propertyType = doc.get("property_type");
        if (propertyType != null) contents.append(propertyType).append(" ");
        
        // 6. Amenities (ya están en contents si fueron indexadas, pero no stored)
        // No podemos recuperarlas si no están stored
        
        // 7. Bathrooms
        String bathrooms = doc.get("bathrooms");
        if (bathrooms != null) contents.append(bathrooms).append(" bathrooms ");
        String bathroomsText = doc.get("bathrooms_text");
        if (bathroomsText != null) contents.append(bathroomsText).append(" ");
        
        // 8. Bedrooms
        String bedrooms = doc.get("bedrooms");
        if (bedrooms != null) contents.append(bedrooms).append(" bedrooms ");
        
        // 9. Price
        String price = doc.get("price");
        if (price != null) contents.append("price ").append(price).append(" ");
        
        // 10. Number of reviews
        String numReviews = doc.get("number_of_reviews");
        if (numReviews != null) contents.append(numReviews).append(" reviews ");
        
        // 11. Review Scores Rating
        String rating = doc.get("review_scores_rating");
        if (rating != null) contents.append("rating ").append(rating).append(" ");
        
        String result = contents.toString();
        // Guardar en cache usando la clave generada
        contentsCache.put(cacheKey, result);
        return result;
    }

    /**
     * Crea un índice temporal completo con todos los documentos válidos (para usar con DatasetSplitter)
     */
    private Directory crearIndiceTemporalCompleto(IndexReader fullReader, List<Integer> docIds,
            Analyzer analyzer, Similarity similarity, String classField, String textField) throws IOException {
        Directory dir = new ByteBuffersDirectory();
        org.apache.lucene.index.IndexWriterConfig iwc = new org.apache.lucene.index.IndexWriterConfig(analyzer);
        iwc.setSimilarity(similarity);
        iwc.setRAMBufferSizeMB(256.0);
        org.apache.lucene.index.IndexWriter writer = new org.apache.lucene.index.IndexWriter(dir, iwc);

        IndexSearcher searcher = new IndexSearcher(fullReader);
        for (int docId : docIds) {
            Document originalDoc = searcher.storedFields().document(docId);
            Document newDoc = new Document();
            
            String classValue = originalDoc.get(classField);
            if (classValue != null && !classValue.trim().isEmpty()) {
                String normalizedClassValue = classValue.toLowerCase().trim();
                newDoc.add(new org.apache.lucene.document.StringField(classField,
                        normalizedClassValue, org.apache.lucene.document.Field.Store.YES));
                newDoc.add(new org.apache.lucene.document.SortedDocValuesField(classField,
                        new BytesRef(normalizedClassValue)));
            }
            
            String contents = reconstruirContents(originalDoc);
            newDoc.add(new org.apache.lucene.document.TextField(textField, contents,
                    org.apache.lucene.document.Field.Store.YES));
            
            writer.addDocument(newDoc);
        }

        writer.commit();
        writer.close();
        return dir;
    }

    /**
     * Crea un índice temporal completo con documentos y categorías (para usar con DatasetSplitter)
     */
    private Directory crearIndiceTemporalCompletoConCategoria(List<DocumentWithClass> docs, 
            Analyzer analyzer, Similarity similarity, String classField) throws IOException {
        Directory dir = new ByteBuffersDirectory();
        org.apache.lucene.index.IndexWriterConfig iwc = new org.apache.lucene.index.IndexWriterConfig(analyzer);
        iwc.setSimilarity(similarity);
        iwc.setRAMBufferSizeMB(256.0);
        org.apache.lucene.index.IndexWriter writer = new org.apache.lucene.index.IndexWriter(dir, iwc);

        for (DocumentWithClass docWithClass : docs) {
            Document newDoc = new Document();
            
            newDoc.add(new org.apache.lucene.document.StringField(classField,
                    docWithClass.category, org.apache.lucene.document.Field.Store.YES));
            newDoc.add(new org.apache.lucene.document.SortedDocValuesField(classField,
                    new BytesRef(docWithClass.category)));
            
            String contents = reconstruirContents(docWithClass.doc);
            newDoc.add(new org.apache.lucene.document.TextField(FIELD_CONTENTS, contents,
                    org.apache.lucene.document.Field.Store.YES));
            
            writer.addDocument(newDoc);
        }

        writer.commit();
        writer.close();
        return dir;
    }

    /**
     * Crea un índice temporal con un subconjunto de documentos del reader original
     * @deprecated Usar crearIndiceTemporalCompleto + DatasetSplitter en su lugar
     */
    @Deprecated
    private IndexReader crearIndiceTemporalDesdeIds(IndexReader fullReader, List<Integer> docIds, 
            Analyzer analyzer, Similarity similarity, String classField, String textField) throws IOException {
        // Usar ByteBuffersDirectory para índices temporales (mejor rendimiento en memoria)
        Directory dir = new ByteBuffersDirectory();
        org.apache.lucene.index.IndexWriterConfig iwc = new org.apache.lucene.index.IndexWriterConfig(analyzer);
        iwc.setSimilarity(similarity);
        // Aumentar buffer del IndexWriter para mejor rendimiento
        iwc.setRAMBufferSizeMB(256.0);
        org.apache.lucene.index.IndexWriter writer = new org.apache.lucene.index.IndexWriter(dir, iwc);

        IndexSearcher searcher = new IndexSearcher(fullReader);
        for (int docId : docIds) {
            Document originalDoc = searcher.storedFields().document(docId);
            Document newDoc = new Document();
            
            // Solo copiar campos necesarios: classField y textField/contents
            // Obtener valor de clase - el campo ya está normalizado en el índice original
            // Leer directamente del campo normalizado (no del _original)
            String classValue = originalDoc.get(classField);
            if (classValue != null && !classValue.trim().isEmpty()) {
                // El valor ya está normalizado a lowercase en el indexador
                // Solo necesitamos asegurarnos de que esté limpio
                String normalizedClassValue = classValue.toLowerCase().trim();
                
                // IMPORTANTE: No mapear los valores aquí porque el BM25NBClassifier
                // necesita leer las clases exactamente como están en el índice original
                // El mapeo podría estar causando que el clasificador no encuentre las clases correctas
                
                // Agregar campo de clase como StringField y SortedDocValuesField
                // Usar el valor exacto del índice original para consistencia
                newDoc.add(new org.apache.lucene.document.StringField(classField,
                        normalizedClassValue, org.apache.lucene.document.Field.Store.YES));
                newDoc.add(new org.apache.lucene.document.SortedDocValuesField(classField,
                        new BytesRef(normalizedClassValue)));
            }
            
            // Reconstruir contents desde campos stored disponibles (usando cache)
            String contents = reconstruirContents(originalDoc);
            // Agregar contents como stored para que los clasificadores puedan usarlo
            newDoc.add(new org.apache.lucene.document.TextField(textField, contents,
                    org.apache.lucene.document.Field.Store.YES));
            
            writer.addDocument(newDoc);
        }

        writer.commit();
        writer.close();

        return DirectoryReader.open(dir);
    }

    /**
     * Crea un índice temporal con los documentos y sus categorías
     * @deprecated Usar crearIndiceTemporalCompletoConCategoria + DatasetSplitter en su lugar
     */
    @Deprecated
    private IndexReader crearIndiceTemporal(List<DocumentWithClass> docs, Analyzer analyzer, Similarity similarity,
            String classField) throws IOException {
        // Usar ByteBuffersDirectory para índices temporales (mejor rendimiento en memoria)
        Directory dir = new ByteBuffersDirectory();
        org.apache.lucene.index.IndexWriterConfig iwc = new org.apache.lucene.index.IndexWriterConfig(analyzer);
        iwc.setSimilarity(similarity);
        // Aumentar buffer del IndexWriter para mejor rendimiento
        iwc.setRAMBufferSizeMB(256.0);
        org.apache.lucene.index.IndexWriter writer = new org.apache.lucene.index.IndexWriter(dir, iwc);

        for (DocumentWithClass docWithClass : docs) {
            Document newDoc = new Document();
            
            // Solo copiar campos necesarios: classField y contents
            // Agregar campo de categoría (puede ser property_type_category o bedrooms_category)
            // Los clasificadores necesitan que el campo de clase esté indexado como StringField
            // y también como SortedDocValuesField para que puedan leerlo correctamente
            newDoc.add(new org.apache.lucene.document.StringField(classField,
                    docWithClass.category, org.apache.lucene.document.Field.Store.YES));
            newDoc.add(new org.apache.lucene.document.SortedDocValuesField(classField,
                    new BytesRef(docWithClass.category)));
            
            // Reconstruir contents desde campos stored disponibles (usando cache)
            String contents = reconstruirContents(docWithClass.doc);
            // Agregar contents como stored para que los clasificadores puedan usarlo
            newDoc.add(new org.apache.lucene.document.TextField(FIELD_CONTENTS, contents,
                    org.apache.lucene.document.Field.Store.YES));
            
            writer.addDocument(newDoc);
        }

        writer.commit();
        writer.close();

        return DirectoryReader.open(dir);
    }

    /**
     * Evalúa un clasificador y retorna métricas usando ConfusionMatrixGenerator de Lucene
     */
    private EvaluationResults evaluarClasificador(Classifier<BytesRef> classifier, IndexReader testReader,
            String classField, String textField, String classifierName) throws Exception {
        EvaluationResults results = new EvaluationResults(classifierName);

        // Logs de depuración para BM25NBClassifier (recopilar antes de usar ConfusionMatrixGenerator)
        boolean isBM25NB = classifierName.contains("BM25NBClassifier");
        Map<String, Integer> predictionDistribution = new HashMap<>();
        int debugSampleCount = 0;
        int maxDebugSamples = 10;
        List<String> debugSamples = new ArrayList<>();
        
        if (isBM25NB) {
            // Recopilar información de debug antes de generar la matriz de confusión
            IndexSearcher searcher = new IndexSearcher(testReader);
            TopDocs topDocs = searcher.search(new MatchAllDocsQuery(), Math.min(testReader.numDocs(), 100));
            
            for (ScoreDoc scoreDoc : topDocs.scoreDocs) {
                Document doc = searcher.storedFields().document(scoreDoc.doc);
                String text = doc.get(textField);
                if (text == null || text.trim().isEmpty()) {
                    text = reconstruirContents(doc);
                }
                
                if (text != null && !text.trim().isEmpty()) {
                    try {
                        ClassificationResult<BytesRef> result = classifier.assignClass(text);
                        if (result != null && result.assignedClass() != null) {
                            String predictedClass = result.assignedClass().utf8ToString().toLowerCase().trim();
                            predictionDistribution.put(predictedClass, 
                                predictionDistribution.getOrDefault(predictedClass, 0) + 1);
                            
                            if (debugSampleCount < maxDebugSamples) {
                                String actualClass = doc.get(classField);
                                if (actualClass != null) actualClass = actualClass.toLowerCase().trim();
                                
                                double score = 0.0;
                                try {
                                    java.lang.reflect.Method scoreMethod = result.getClass().getMethod("score");
                                    Object scoreObj = scoreMethod.invoke(result);
                                    if (scoreObj instanceof Number) {
                                        score = ((Number) scoreObj).doubleValue();
                                    }
                                } catch (Exception e) {
                                    score = -1.0;
                                }
                                
                                String sampleText = text.length() > 100 ? text.substring(0, 100) + "..." : text;
                                String scoreStr = score >= 0 ? String.format("%.4f", score) : "N/A";
                                String debugMsg = String.format(
                                    "  [DEBUG] Doc %d: Real='%s' | Pred='%s' | Score=%s | Text='%s'",
                                    scoreDoc.doc, actualClass, predictedClass, scoreStr, sampleText
                                );
                                debugSamples.add(debugMsg);
                                System.out.println(debugMsg);
                                debugSampleCount++;
                            }
                        }
                    } catch (Exception e) {
                        // Ignorar errores en debug
                    }
                }
            }
        }

        // Usar ConfusionMatrixGenerator de Lucene
        ConfusionMatrixGenerator.ConfusionMatrix confusionMatrix = 
            ConfusionMatrixGenerator.getConfusionMatrix(testReader, classifier, classField, textField, 
                testReader.numDocs());

        // Extraer métricas de la matriz de confusión
        results.accuracy = confusionMatrix.getAccuracy();
        
        // Obtener todas las clases desde la matriz linearizada
        // La matriz es Map<actualClass, Map<predictedClass, count>>
        Map<String, Map<String, Long>> linearizedMatrix = confusionMatrix.getLinearizedMatrix();
        Set<String> allClasses = new HashSet<>();
        for (Map<String, Long> predictedMap : linearizedMatrix.values()) {
            allClasses.addAll(predictedMap.keySet());
        }
        allClasses.addAll(linearizedMatrix.keySet());
        
        // Calcular métricas por clase
        for (String clazz : allClasses) {
            double precision = confusionMatrix.getPrecision(clazz);
            double recall = confusionMatrix.getRecall(clazz);
            double f1 = confusionMatrix.getF1Measure(clazz); // Método correcto: getF1Measure, no getF1
            
            results.precisionByClass.put(clazz, precision);
            results.recallByClass.put(clazz, recall);
            results.f1ByClass.put(clazz, f1);
        }

        // Métricas globales de matriz de confusión
        // Calcular TP, FP, FN, TN desde la matriz linearizada
        // La matriz es: Map<actualClass, Map<predictedClass, count>>
        // IMPORTANTE: En multi-clase, cada documento aparece exactamente una vez en la matriz
        // Cada off-diagonal cell [i][j] donde i != j representa un error que se cuenta como:
        //   - FP para la clase j (predicted class)
        //   - FN para la clase i (actual class)
        // Por lo tanto, totalFP = totalFN = suma de todos los elementos off-diagonal
        
        // Primero, calcular el total de documentos desde la matriz de confusión
        int totalDocsFromMatrix = 0;
        for (Map<String, Long> predictedMap : linearizedMatrix.values()) {
            for (Long count : predictedMap.values()) {
                totalDocsFromMatrix += count.intValue();
            }
        }
        
        // TP = suma de elementos diagonales (actual == predicted)
        int totalTP = 0;
        for (Map.Entry<String, Map<String, Long>> actualEntry : linearizedMatrix.entrySet()) {
            String actualClass = actualEntry.getKey();
            Map<String, Long> predictedMap = actualEntry.getValue();
            Long tp = predictedMap.get(actualClass);
            if (tp != null) {
                totalTP += tp.intValue();
            }
        }
        
        // FP = suma de elementos donde predicted != actual (off-diagonal)
        // Cada off-diagonal cell [i][j] donde i != j contribuye a FP para la clase j
        int totalFP = 0;
        for (Map.Entry<String, Map<String, Long>> actualEntry : linearizedMatrix.entrySet()) {
            String actualClass = actualEntry.getKey();
            Map<String, Long> predictedMap = actualEntry.getValue();
            for (Map.Entry<String, Long> predictedEntry : predictedMap.entrySet()) {
                String predictedClass = predictedEntry.getKey();
                if (!actualClass.equals(predictedClass)) {
                    totalFP += predictedEntry.getValue().intValue();
                }
            }
        }
        
        // FN = totalFP (en multi-clase, cada error se cuenta como FP y FN)
        int totalFN = totalFP;
        
        // Verificación: TP + FP = totalDocs (cada documento está en TP o en FP)
        // totalFN = totalFP, pero para reportar usamos ambos para consistencia con métricas binarias
        int totalDocs = totalDocsFromMatrix;
        int totalTN = 0; // En multi-clase agregado, TN siempre es 0
        
        results.truePositives = totalTP;
        results.falsePositives = totalFP;
        results.falseNegatives = totalFN;
        results.trueNegatives = totalTN;

        // Logs de depuración para BM25NBClassifier
        if (isBM25NB) {
            int total = testReader.numDocs();
            System.out.println("\n  [DEBUG BM25NBClassifier] Resumen de depuración:");
            System.out.println("  - Total documentos clasificados: " + total);
            System.out.println("  - Distribución de predicciones:");
            for (Map.Entry<String, Integer> entry : predictionDistribution.entrySet()) {
                double percentage = total > 0 ? (double) entry.getValue() / total * 100.0 : 0.0;
                System.out.println(String.format("    '%s': %d (%.2f%%)", entry.getKey(), entry.getValue(), percentage));
            }
            System.out.println("  - Distribución de clases reales:");
            Map<String, Integer> actualDistribution = new HashMap<>();
            for (String clazz : allClasses) {
                // Calcular TP y FN para esta clase desde la matriz linearizada
                int tp = 0, fn = 0;
                Map<String, Long> actualRow = linearizedMatrix.get(clazz);
                if (actualRow != null) {
                    Long tpValue = actualRow.get(clazz);
                    if (tpValue != null) {
                        tp = tpValue.intValue();
                    }
                    // FN = suma de todos los elementos donde actual=clazz pero predicted != clazz
                    for (Map.Entry<String, Long> predEntry : actualRow.entrySet()) {
                        if (!predEntry.getKey().equals(clazz)) {
                            fn += predEntry.getValue().intValue();
                        }
                    }
                }
                int count = tp + fn;
                actualDistribution.put(clazz, count);
            }
            for (Map.Entry<String, Integer> entry : actualDistribution.entrySet()) {
                double percentage = total > 0 ? (double) entry.getValue() / total * 100.0 : 0.0;
                System.out.println(String.format("    '%s': %d (%.2f%%)", entry.getKey(), entry.getValue(), percentage));
            }
            System.out.println("  - Matriz de confusión por clase:");
            for (String clazz : allClasses) {
                // Calcular TP, FP, FN, TN para esta clase
                int tp = 0, fp = 0, fn = 0;
                
                // TP: actual=clazz y predicted=clazz
                Map<String, Long> actualRow = linearizedMatrix.get(clazz);
                if (actualRow != null) {
                    Long tpValue = actualRow.get(clazz);
                    if (tpValue != null) {
                        tp = tpValue.intValue();
                    }
                    // FN: actual=clazz pero predicted != clazz
                    for (Map.Entry<String, Long> predEntry : actualRow.entrySet()) {
                        if (!predEntry.getKey().equals(clazz)) {
                            fn += predEntry.getValue().intValue();
                        }
                    }
                }
                
                // FP: predicted=clazz pero actual != clazz
                for (Map.Entry<String, Map<String, Long>> actualEntry : linearizedMatrix.entrySet()) {
                    if (!actualEntry.getKey().equals(clazz)) {
                        Map<String, Long> predMap = actualEntry.getValue();
                        Long fpValue = predMap.get(clazz);
                        if (fpValue != null) {
                            fp += fpValue.intValue();
                        }
                    }
                }
                
                // TN = total - TP - FP - FN
                int tn = total - tp - fp - fn;
                System.out.println(String.format("    '%s': TP=%d, FP=%d, FN=%d, TN=%d", clazz, tp, fp, fn, tn));
            }
            if (!debugSamples.isEmpty() && debugSamples.size() > maxDebugSamples) {
                System.out.println("  - Ejemplos adicionales de errores:");
                for (int i = maxDebugSamples; i < debugSamples.size() && i < maxDebugSamples + 5; i++) {
                    System.out.println(debugSamples.get(i));
                }
            }
            System.out.println();
        }

        // Mostrar resultados
        System.out.println("Accuracy: " + String.format("%.4f", results.accuracy));
        System.out.println("TP: " + results.truePositives + ", FP: " + results.falsePositives + 
            ", FN: " + results.falseNegatives + ", TN: " + results.trueNegatives);

        return results;
    }

    /**
     * Muestra tabla comparativa de resultados
     */
    private void mostrarTablaComparativa(List<EvaluationResults> results, String titulo) {
        System.out.println("\n" + "=".repeat(100));
        System.out.println(titulo);
        System.out.println("=".repeat(100));

        if (results.isEmpty()) {
            System.out.println("No hay resultados para mostrar");
            return;
        }

        // Obtener todas las clases de todos los resultados
        Set<String> allClasses = new HashSet<>();
        for (EvaluationResults r : results) {
            allClasses.addAll(r.precisionByClass.keySet());
        }
        List<String> sortedClasses = new ArrayList<>(allClasses);
        Collections.sort(sortedClasses);

        // Función para abreviar nombres de clases largos
        java.util.function.Function<String, String> abbreviate = (clazz) -> {
            // Mapeo de nombres completos a abreviaciones
            if (clazz.contains("city of los angeles")) return "city_la";
            if (clazz.contains("other cities")) return "other";
            if (clazz.contains("unincorporated")) return "unincorp";
            if (clazz.equals("0")) return "0";
            if (clazz.equals("1")) return "1";
            if (clazz.equals("2")) return "2";
            if (clazz.equals("3")) return "3";
            if (clazz.equals("4")) return "4";
            if (clazz.equals("5+")) return "5+";
            // Para otras clases, usar primeros 6 caracteres sin espacios
            return clazz.replaceAll("\\s+", "_").substring(0, Math.min(6, clazz.length()));
        };
        
        // Encabezado
        System.out.printf("%-30s", "Clasificador");
        System.out.printf("%12s", "Accuracy");
        for (String clazz : sortedClasses) {
            String abbrev = abbreviate.apply(clazz);
            System.out.printf("%10s", "P_" + abbrev);
            System.out.printf("%10s", "R_" + abbrev);
            System.out.printf("%10s", "F1_" + abbrev);
        }
        System.out.printf("%8s", "TP");
        System.out.printf("%8s", "FP");
        System.out.printf("%8s", "FN");
        System.out.printf("%8s", "TN");
        System.out.println();

        // Filas
        for (EvaluationResults r : results) {
            System.out.printf("%-30s", r.classifierName);
            System.out.printf("%12.4f", r.accuracy);

            for (String clazz : sortedClasses) {
                double p = r.precisionByClass.getOrDefault(clazz, 0.0);
                double rec = r.recallByClass.getOrDefault(clazz, 0.0);
                double f1 = r.f1ByClass.getOrDefault(clazz, 0.0);
                System.out.printf("%10.4f", p);
                System.out.printf("%10.4f", rec);
                System.out.printf("%10.4f", f1);
            }

            System.out.printf("%8d", r.truePositives);
            System.out.printf("%8d", r.falsePositives);
            System.out.printf("%8d", r.falseNegatives);
            System.out.printf("%8d", r.trueNegatives);
            System.out.println();
        }

        System.out.println("=".repeat(100));
    }
}

