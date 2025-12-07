import javafx.application.Application;
import javafx.application.Platform;
import javafx.beans.property.SimpleDoubleProperty;
import javafx.beans.property.SimpleIntegerProperty;
import javafx.beans.property.SimpleStringProperty;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.geometry.Insets;
import javafx.geometry.Orientation;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.control.cell.PropertyValueFactory;
import javafx.scene.layout.*;
import javafx.stage.Stage;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.DoublePoint;
import org.apache.lucene.facet.DrillDownQuery;
import org.apache.lucene.facet.DrillSideways;
import org.apache.lucene.facet.FacetResult;
import org.apache.lucene.facet.Facets;
import org.apache.lucene.facet.FacetsConfig;
import org.apache.lucene.facet.FacetsCollector;
import org.apache.lucene.facet.LabelAndValue;
import org.apache.lucene.facet.taxonomy.FastTaxonomyFacetCounts;
import org.apache.lucene.facet.taxonomy.TaxonomyReader;
import org.apache.lucene.facet.taxonomy.directory.DirectoryTaxonomyReader;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.similarities.Similarity;
import org.apache.lucene.store.FSDirectory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Sencillo front-end JavaFX para búsquedas sobre el índice de propiedades.
 *
 * Objetivos del primer prototipo:
 * - Barra de búsqueda simple (mega campo "contents").
 * - Sección de búsqueda avanzada muy ligera (rango de precio, barrio).
 * - Panel lateral donde en el futuro se podrán añadir facetas.
 * - Tabla central con resultados básicos.
 *
 * NOTA: este prototipo evita lógica compleja y reusa utilidades de indexación
 * siempre que es posible (analizador, similarity, rutas de índice).
 */
public class AirbnbSearchApp extends Application {

    /**
     * Modelo ligero para representar un resultado en la tabla.
     * Se separa del Document de Lucene para mantener la UI desacoplada.
     */
    public static class PropertyResult {
        private final SimpleIntegerProperty id = new SimpleIntegerProperty();
        private final SimpleStringProperty name = new SimpleStringProperty();
        private final SimpleStringProperty neighbourhood = new SimpleStringProperty();
        private final SimpleStringProperty propertyType = new SimpleStringProperty();
        private final SimpleDoubleProperty price = new SimpleDoubleProperty();

        public PropertyResult(int id, String name, String neighbourhood, String propertyType, Double price) {
            this.id.set(id);
            this.name.set(name != null ? name : "");
            this.neighbourhood.set(neighbourhood != null ? neighbourhood : "");
            this.propertyType.set(propertyType != null ? propertyType : "");
            this.price.set(price != null ? price : 0.0);
        }

        public int getId() {
            return id.get();
        }

        public String getName() {
            return name.get();
        }

        public String getNeighbourhood() {
            return neighbourhood.get();
        }

        public String getPropertyType() {
            return propertyType.get();
        }

        public double getPrice() {
            return price.get();
        }
    }

    // Parámetros básicos: se podría exponer como argumento de línea de comandos
    private String indexRoot = "./index_root";

    // Componentes UI principales
    private TextField simpleQueryField;
    private TextField neighbourhoodField;
    private TextField minPriceField;
    private TextField maxPriceField;
    private Label statusLabel;
    // Contenedor para grupos de facetas (similar a la columna de filtros de un buscador web)
    private VBox facetsContainer;
    private TableView<PropertyResult> resultsTable;
    private final ObservableList<PropertyResult> resultsData = FXCollections.observableArrayList();

    @Override
    public void start(Stage primaryStage) {
        // Permitir pasar indexRoot por argumentos JavaFX (módulo simple)
        Parameters params = getParameters();
        List<String> rawArgs = params.getRaw();
        for (int i = 0; i < rawArgs.size(); i++) {
            String arg = rawArgs.get(i);
            if ("--index-root".equals(arg) && i + 1 < rawArgs.size()) {
                indexRoot = rawArgs.get(i + 1);
            }
        }

        BorderPane root = new BorderPane();
        root.setPadding(new Insets(10));

        root.setTop(createSearchHeader());
        root.setLeft(createFacetSidebar());
        root.setCenter(createResultsTable());
        root.setBottom(createStatusBar());

        Scene scene = new Scene(root, 1100, 700);
        primaryStage.setTitle("Airbnb Lucene Search (JavaFX Prototype)");
        primaryStage.setScene(scene);
        primaryStage.show();
    }

    /**
     * Crea la parte superior: búsqueda simple + búsqueda avanzada plegable.
     */
    private VBox createSearchHeader() {
        VBox header = new VBox(8);
        header.setPadding(new Insets(0, 0, 10, 0));

        // Fila de búsqueda simple
        HBox simpleRow = new HBox(8);
        simpleRow.setAlignment(Pos.CENTER_LEFT);

        simpleQueryField = new TextField();
        simpleQueryField.setPromptText("Buscar en mega campo (ej: 'pool AND 3 bedrooms')");
        simpleQueryField.setOnAction(e -> executeSearch()); // Enter lanza búsqueda

        Button searchButton = new Button("Buscar");
        searchButton.setDefaultButton(true);
        searchButton.setOnAction(e -> executeSearch());

        Button clearButton = new Button("Reset");
        clearButton.setOnAction(e -> clearFiltersAndResults());

        simpleRow.getChildren().addAll(new Label("Consulta:"), simpleQueryField, searchButton, clearButton);
        HBox.setHgrow(simpleQueryField, Priority.ALWAYS);

        // Sección de búsqueda avanzada (muy sencilla en este prototipo)
        GridPane advancedGrid = new GridPane();
        advancedGrid.setHgap(8);
        advancedGrid.setVgap(4);
        advancedGrid.setPadding(new Insets(5, 0, 0, 0));

        neighbourhoodField = new TextField();
        neighbourhoodField.setPromptText("neighbourhood_cleansed (ej: hollywood)");

        minPriceField = new TextField();
        minPriceField.setPromptText("Precio mín.");
        maxPriceField = new TextField();
        maxPriceField.setPromptText("Precio máx.");

        advancedGrid.add(new Label("Barrio:"), 0, 0);
        advancedGrid.add(neighbourhoodField, 1, 0);
        advancedGrid.add(new Label("Precio:"), 0, 1);
        HBox priceBox = new HBox(5, minPriceField, new Label("-"), maxPriceField);
        advancedGrid.add(priceBox, 1, 1);
        GridPane.setHgrow(neighbourhoodField, Priority.ALWAYS);

        TitledPane advancedPane = new TitledPane("Búsqueda avanzada", advancedGrid);
        advancedPane.setExpanded(false);

        header.getChildren().addAll(simpleRow, advancedPane);
        return header;
    }

    /**
     * Crea un sidebar para facetas con estructura similar a la de un buscador web:
     * botones de "Aplicar/Quitar filtros" y bloques plegables por categoría.
     */
    private VBox createFacetSidebar() {
        VBox side = new VBox(8);
        side.setPadding(new Insets(0, 10, 0, 0));
        side.setPrefWidth(280);

        // Fila de botones estilo "Aplicar filtros" / "Quitar filtros"
        HBox buttonsRow = new HBox(8);
        buttonsRow.setAlignment(Pos.CENTER);
        Button applyButton = new Button("Aplicar filtros");
        applyButton.setOnAction(e -> applyFacetSelection());
        Button clearButton = new Button("Quitar filtros");
        clearButton.setOnAction(e -> clearFacetSelection());
        buttonsRow.getChildren().addAll(applyButton, clearButton);

        Label title = new Label("Filtros por facetas");
        title.setStyle("-fx-font-weight: bold;");

        facetsContainer = new VBox(6);
        facetsContainer.setPadding(new Insets(6, 0, 0, 0));
        facetsContainer.getChildren().add(new Label("Ejecuta una búsqueda para ver facetas disponibles."));

        ScrollPane scroll = new ScrollPane(facetsContainer);
        scroll.setFitToWidth(true);
        scroll.setHbarPolicy(ScrollPane.ScrollBarPolicy.NEVER);

        side.getChildren().addAll(buttonsRow, new Separator(), title, scroll);
        VBox.setVgrow(scroll, Priority.ALWAYS);

        return side;
    }

    /**
     * Crea la tabla central de resultados.
     */
    private TableView<PropertyResult> createResultsTable() {
        resultsTable = new TableView<>();
        resultsTable.setItems(resultsData);

        TableColumn<PropertyResult, Integer> idCol = new TableColumn<>("ID");
        idCol.setCellValueFactory(new PropertyValueFactory<>("id"));
        idCol.setPrefWidth(70);

        TableColumn<PropertyResult, String> nameCol = new TableColumn<>("Nombre");
        nameCol.setCellValueFactory(new PropertyValueFactory<>("name"));
        nameCol.setPrefWidth(320);

        TableColumn<PropertyResult, String> neighCol = new TableColumn<>("Barrio");
        neighCol.setCellValueFactory(new PropertyValueFactory<>("neighbourhood"));
        neighCol.setPrefWidth(160);

        TableColumn<PropertyResult, String> typeCol = new TableColumn<>("Tipo");
        typeCol.setCellValueFactory(new PropertyValueFactory<>("propertyType"));
        typeCol.setPrefWidth(140);

        TableColumn<PropertyResult, Double> priceCol = new TableColumn<>("Precio");
        priceCol.setCellValueFactory(new PropertyValueFactory<>("price"));
        priceCol.setPrefWidth(100);

        resultsTable.getColumns().addAll(idCol, nameCol, neighCol, typeCol, priceCol);

        return resultsTable;
    }

    /**
     * Barra de estado inferior.
     */
    private HBox createStatusBar() {
        HBox bar = new HBox(10);
        bar.setAlignment(Pos.CENTER_LEFT);
        bar.setPadding(new Insets(8, 0, 0, 0));

        statusLabel = new Label("Listo.");
        Label queryLabel = new Label();
        queryLabel.setStyle("-fx-font-size: 11px; -fx-text-fill: #666;");

        // Guardamos la referencia en la propia etiqueta de estado mediante userData para no
        // añadir más campos a la clase.
        statusLabel.setUserData(queryLabel);

        bar.getChildren().addAll(statusLabel, new Separator(Orientation.VERTICAL), queryLabel);

        return bar;
    }

    /**
     * Ejecuta la búsqueda en un hilo de fondo sencillo y actualiza la UI al terminar.
     * Para mantenerlo KISS, se usan clases básicas de Lucene en lugar de reutilizar
     * el menú interactivo de BusquedasLucene.
     */
    private void executeSearch() {
        final String queryText = simpleQueryField.getText() != null
                ? simpleQueryField.getText().trim()
                : "";
        final String neighbourhood = neighbourhoodField.getText() != null
                ? neighbourhoodField.getText().trim().toLowerCase()
                : "";
        final String minPriceText = minPriceField.getText() != null ? minPriceField.getText().trim() : "";
        final String maxPriceText = maxPriceField.getText() != null ? maxPriceField.getText().trim() : "";

        statusLabel.setText("Buscando...");

        // Sin facetas activas al pulsar "Buscar"
        runSearchInBackground(queryText, neighbourhood, minPriceText, maxPriceText, null);
    }

    /**
     * Ejecuta la búsqueda (con o sin faceta) en un hilo de fondo.
     * Si facetDim/facetLabel son no nulos, se aplica DrillDownQuery como en
     * BusquedasLucene.ejecutarQueryMegaCampoConFacetas (7.2).
     */
    private void runSearchInBackground(String queryText,
                                       String neighbourhood,
                                       String minPriceText,
                                       String maxPriceText,
                                       Map<String, List<String>> activeFacets) {
        new Thread(() -> {
            long start = System.currentTimeMillis();
            List<PropertyResult> results = new ArrayList<>();
            Map<String, List<LabelAndValue>> facetsData = new LinkedHashMap<>();
            String luceneQueryText = "";
            String errorMessage = null;

            try {
                Analyzer analyzer = AirbnbIndexador.crearAnalizador();
                Similarity similarity = AirbnbIndexador.crearSimilarity();

                IndexReader reader = DirectoryReader.open(
                        FSDirectory.open(AirbnbIndexador.getPropertiesIndexPath(indexRoot)));
                IndexSearcher searcher = new IndexSearcher(reader);
                searcher.setSimilarity(similarity);

                BooleanQuery.Builder queryBuilder = new BooleanQuery.Builder();

                // 1) Consulta libre sobre mega campo "contents"
                if (queryText != null && !queryText.isEmpty()) {
                    QueryParser parser = new QueryParser(AirbnbIndexador.FIELD_CONTENTS, analyzer);
                    Query q = parser.parse(queryText);
                    queryBuilder.add(q, BooleanClause.Occur.MUST);
                } else {
                    // Si no hay texto, usamos MatchAllDocsQuery para poder aplicar solo filtros
                    queryBuilder.add(new MatchAllDocsQuery(), BooleanClause.Occur.MUST);
                }

                // 2) Filtro por neighbourhood_cleansed (StringField normalizada a lowercase)
                if (neighbourhood != null && !neighbourhood.isEmpty()) {
                    Query neighQuery = new TermQuery(new Term("neighbourhood_cleansed", neighbourhood));
                    queryBuilder.add(neighQuery, BooleanClause.Occur.FILTER);
                }

                // 3) Filtros por precio (DoublePoint)
                Double minPrice = parseDoubleOrNull(minPriceText);
                Double maxPrice = parseDoubleOrNull(maxPriceText);
                if (minPrice != null || maxPrice != null) {
                    double min = minPrice != null ? minPrice : 0.0;
                    double max = maxPrice != null ? maxPrice : Double.MAX_VALUE;
                    Query priceQuery = DoublePoint.newRangeQuery("price", min, max);
                    queryBuilder.add(priceQuery, BooleanClause.Occur.FILTER);
                }

                Query baseQuery = queryBuilder.build();

                try (TaxonomyReader taxoReader = new DirectoryTaxonomyReader(
                        FSDirectory.open(AirbnbIndexador.getTaxoPropertiesIndexPath(indexRoot)))) {
                    // IMPORTANTE: createFacetsConfig() ya tiene config.setHierarchical("property_type", true)
                    // Esto es necesario para que Lucene trate property_type como jerárquico
                    FacetsConfig fconfig = AirbnbIndexador.createFacetsConfig();
                    
                    // Primero obtener todas las facetas disponibles para poder expandir categorías
                    FacetsCollector fcInitial = new FacetsCollector();
                    searcher.search(baseQuery, fcInitial);
                    Facets facetsInitial = new FastTaxonomyFacetCounts(taxoReader, fconfig, fcInitial);
                    List<FacetResult> allDimsInitial = facetsInitial.getAllDims(20);
                    Map<String, List<LabelAndValue>> availableFacets = new LinkedHashMap<>();
                    if (allDimsInitial != null) {
                        for (FacetResult fr : allDimsInitial) {
                            if (fr != null && fr.dim != null) {
                                availableFacets.put(fr.dim, Arrays.asList(fr.labelValues));
                            }
                        }
                    }
                    
                    // Si hay categorías padre seleccionadas, necesitamos obtener TODAS las facetas del índice completo
                    // para poder expandir correctamente las categorías padre
                    Map<String, List<LabelAndValue>> allFacetsFromIndex = null;
                    if (activeFacets != null && activeFacets.containsKey("property_type")) {
                        List<String> propertyTypeSelections = activeFacets.get("property_type");
                        boolean hasParentCategory = false;
                        for (String label : propertyTypeSelections) {
                            if (!label.contains("/")) {
                                // Es una categoría padre
                                hasParentCategory = true;
                                break;
                            }
                        }
                        
                        if (hasParentCategory) {
                            // Obtener todas las facetas del índice completo (sin filtros)
                            FacetsCollector fcAll = new FacetsCollector();
                            searcher.search(new MatchAllDocsQuery(), fcAll);
                            Facets facetsAll = new FastTaxonomyFacetCounts(taxoReader, fconfig, fcAll);
                            List<FacetResult> allDimsAll = facetsAll.getAllDims(1000); // Obtener muchas para asegurar que tenemos todas
                            allFacetsFromIndex = new LinkedHashMap<>();
                            if (allDimsAll != null) {
                                for (FacetResult fr : allDimsAll) {
                                    if (fr != null && fr.dim != null) {
                                        allFacetsFromIndex.put(fr.dim, Arrays.asList(fr.labelValues));
                                    }
                                }
                            }
                        }
                    }
                    
                    TopDocs topDocs;
                    
                    // Si hay facetas seleccionadas, usamos DrillSideways para mantener conteos de todas las facetas
                    if (activeFacets != null && !activeFacets.isEmpty()) {
                        DrillDownQuery ddq = new DrillDownQuery(fconfig, baseQuery);
                        for (Map.Entry<String, List<String>> entry : activeFacets.entrySet()) {
                            String dim = entry.getKey();
                            
                            // Para property_type jerárquico, usar la API de jerarquía de Lucene correctamente
                            if ("property_type".equals(dim)) {
                                // Separar categorías principales de subfacetas
                                Set<String> categoriesSelected = new HashSet<>();
                                Map<String, List<String>> categoryGroups = new HashMap<>();
                                
                                for (String label : entry.getValue()) {
                                    if (label.contains("/")) {
                                        // Path completo: "home/entire home"
                                        String[] parts = label.split("/", 2);
                                        if (parts.length == 2) {
                                            String category = parts[0];
                                            String subType = parts[1];
                                            categoryGroups.computeIfAbsent(category, k -> new ArrayList<>()).add(subType);
                                        }
                                    } else {
                                        // Categoría principal seleccionada directamente: "home"
                                        categoriesSelected.add(label);
                                    }
                                }
                                
                                // Procesar cada categoría:
                                // - Si tiene hijos específicos seleccionados, usar solo esos hijos
                                // - Si NO tiene hijos específicos pero está seleccionada como padre, usar TODOS sus hijos
                                for (String category : categoriesSelected) {
                                    if (categoryGroups.containsKey(category)) {
                                        // La categoría padre está seleccionada Y tiene hijos específicos seleccionados
                                        // Usar SOLO los hijos específicos seleccionados (no todos los hijos del padre)
                                        List<String> subTypes = categoryGroups.get(category);
                                        for (String subType : subTypes) {
                                            String fullPath = category + "/" + subType;
                                            ddq.add(dim.trim(), fullPath);
                                        }
                                    } else {
                                        // La categoría padre está seleccionada pero NO tiene hijos específicos seleccionados
                                        // Usar TODOS los hijos del padre del índice completo
                                        List<LabelAndValue> propertyTypeFacets = (allFacetsFromIndex != null) 
                                            ? allFacetsFromIndex.get("property_type") 
                                            : availableFacets.get("property_type");
                                        
                                        if (propertyTypeFacets != null) {
                                            for (LabelAndValue lv : propertyTypeFacets) {
                                                String path = lv.label;
                                                if (path.startsWith(category + "/")) {
                                                    // Esta es una subfaceta de la categoría seleccionada
                                                    // Agregar el path completo como un solo string (como en BusquedasLucene)
                                                    ddq.add(dim.trim(), path);
                                                }
                                            }
                                        }
                                    }
                                }
                                
                                // Agregar subfacetas específicas de categorías que NO están seleccionadas como padre
                                // (solo hijos específicos sin el padre)
                                for (Map.Entry<String, List<String>> catEntry : categoryGroups.entrySet()) {
                                    String category = catEntry.getKey();
                                    // Solo agregar si la categoría NO está en categoriesSelected (solo hijos, sin padre)
                                    if (!categoriesSelected.contains(category)) {
                                        List<String> subTypes = catEntry.getValue();
                                        for (String subType : subTypes) {
                                            // Construir el path completo y agregarlo como un solo string
                                            String fullPath = category + "/" + subType;
                                            ddq.add(dim.trim(), fullPath);
                                        }
                                    }
                                }
                            } else {
                                // Faceta plana (no jerárquica): pasar el valor directamente
                                for (String label : entry.getValue()) {
                                    ddq.add(dim.trim(), label.trim());
                                }
                            }
                        }
                        
                        // Usar DrillSideways para mantener conteos de facetas relacionadas
                        DrillSideways drillSideways = new DrillSideways(searcher, fconfig, taxoReader);
                        DrillSideways.DrillSidewaysResult dsResult = drillSideways.search(ddq, 50);
                        
                        topDocs = dsResult.hits;
                        
                        // Obtener facetas del resultado de DrillSideways (mantiene conteos de todas las facetas)
                        Facets facets = dsResult.facets;
                        List<FacetResult> allDims = facets.getAllDims(20);
                        if (allDims != null) {
                            for (FacetResult fr : allDims) {
                                if (fr != null && fr.dim != null) {
                                    facetsData.put(fr.dim, Arrays.asList(fr.labelValues));
                                }
                            }
                        }
                    } else {
                        // Sin facetas activas: búsqueda normal y recolección de facetas estándar
                        topDocs = searcher.search(baseQuery, 50);
                        facetsData = availableFacets;
                    }

                    // Procesar resultados de documentos
                    for (ScoreDoc sd : topDocs.scoreDocs) {
                        Document doc = searcher.storedFields().document(sd.doc);

                        int id = 0;
                        // El ID se almacena como IntPoint + StoredField. Recuperamos el StoredField.
                        String idStr = doc.get("id");
                        if (idStr != null) {
                            try {
                                id = (int) Double.parseDouble(idStr);
                            } catch (NumberFormatException ignored) {
                            }
                        }

                        String name = doc.get("name");
                        String neigh = doc.get("neighbourhood_cleansed_original");
                        String type = doc.get("property_type_original");

                        Double price = null;
                        String priceStr = doc.get("price");
                        if (priceStr != null) {
                            try {
                                price = Double.parseDouble(priceStr);
                            } catch (NumberFormatException ignored) {
                            }
                        }

                        results.add(new PropertyResult(id, name, neigh, type, price));
                    }
                }

                // Construir texto de query para mostrar
                if (activeFacets != null && !activeFacets.isEmpty()) {
                    StringBuilder queryTextBuilder = new StringBuilder(baseQuery.toString());
                    queryTextBuilder.append(" [Facetas: ");
                    for (Map.Entry<String, List<String>> entry : activeFacets.entrySet()) {
                        queryTextBuilder.append(entry.getKey()).append("=").append(entry.getValue()).append(" ");
                    }
                    queryTextBuilder.append("]");
                    luceneQueryText = queryTextBuilder.toString();
                } else {
                    luceneQueryText = baseQuery.toString();
                }

                reader.close();

            } catch (Exception e) {
                errorMessage = e.getMessage();
            }

            final String errorFinal = errorMessage;
            final long elapsed = System.currentTimeMillis() - start;
            final String luceneQueryFinal = luceneQueryText;
            final List<PropertyResult> resultsFinal = results;
            final Map<String, List<LabelAndValue>> facetsFinal = facetsData;

            Platform.runLater(() -> {
                resultsData.setAll(resultsFinal);
                rebuildFacetSidebar(facetsFinal, activeFacets);

                // Recuperar el label de la query desde userData de statusLabel
                Label queryLabel = null;
                Object ud = statusLabel.getUserData();
                if (ud instanceof Label) {
                    queryLabel = (Label) ud;
                }

                if (errorFinal != null) {
                    statusLabel.setText("Error en la búsqueda: " + errorFinal);
                    if (queryLabel != null) {
                        queryLabel.setText("");
                    }
                } else {
                    statusLabel.setText("Encontrados " + resultsFinal.size() + " resultados en " + elapsed + " ms.");
                    if (queryLabel != null) {
                        queryLabel.setText("Query Lucene: " + luceneQueryFinal);
                    }
                }
            });
        }, "lucene-search-thread").start();
    }

    private static Double parseDoubleOrNull(String text) {
        if (text == null || text.isBlank()) {
            return null;
        }
        try {
            return Double.parseDouble(text.replace(",", "."));
        } catch (NumberFormatException e) {
            return null;
        }
    }

    public static void main(String[] args) {
        // Pasamos args directamente a JavaFX; esto permite usar --index-root también aquí.
        launch(args);
    }

    /**
     * Limpia todos los campos de búsqueda y resultados (botón Reset).
     */
    private void clearFiltersAndResults() {
        simpleQueryField.clear();
        neighbourhoodField.clear();
        minPriceField.clear();
        maxPriceField.clear();
        resultsData.clear();
        if (facetsContainer != null) {
            facetsContainer.getChildren().clear();
            facetsContainer.getChildren().add(new Label("Ejecuta una búsqueda para ver facetas disponibles."));
        }

        // Reset barra de estado y query mostrada
        statusLabel.setText("Listo. Filtros limpiados.");
        Object ud = statusLabel.getUserData();
        if (ud instanceof Label) {
            ((Label) ud).setText("");
        }
    }

    /**
     * Reconstruye el sidebar de facetas con estructura de bloques
     * plegables, similar a un faceted search típico.
     */
    private void rebuildFacetSidebar(Map<String, List<LabelAndValue>> facetsData,
                                     Map<String, List<String>> activeFacets) {
        if (facetsContainer == null) {
            return;
        }
        facetsContainer.getChildren().clear();

        if (facetsData == null || facetsData.isEmpty()) {
            facetsContainer.getChildren().add(new Label("Sin facetas para esta búsqueda."));
            return;
        }

        // Mapeo de nombres técnicos a nombres amigables en español
        Map<String, String> friendlyNames = new LinkedHashMap<>();
        friendlyNames.put("reviews_range", "Número de Reseñas");
        friendlyNames.put("price_range", "Rango de Precio");
        friendlyNames.put("rating_range", "Calificación (Estrellas)");
        friendlyNames.put("neighbourhood_cleansed", "Barrio");
        friendlyNames.put("property_type", "Tipo de Propiedad");
        
        // Mapeo de categorías principales de property_type a nombres amigables
        Map<String, String> propertyTypeCategoryNames = new HashMap<>();
        propertyTypeCategoryNames.put("home", "Home");
        propertyTypeCategoryNames.put("condo", "Condo");
        propertyTypeCategoryNames.put("villa", "Villa");
        propertyTypeCategoryNames.put("guesthouse", "Guesthouse");
        propertyTypeCategoryNames.put("guest suite", "Guest Suite");
        propertyTypeCategoryNames.put("hotel", "Hotel");
        propertyTypeCategoryNames.put("bungalow", "Bungalow");
        propertyTypeCategoryNames.put("townhouse", "Townhouse");
        propertyTypeCategoryNames.put("loft", "Loft");
        propertyTypeCategoryNames.put("serviced apartment", "Serviced Apartment");
        propertyTypeCategoryNames.put("cabin", "Cabin");
        propertyTypeCategoryNames.put("cottage", "Cottage");
        propertyTypeCategoryNames.put("rental unit", "Rental Unit");
        propertyTypeCategoryNames.put("other", "Otros");
        
        // Mapeo de etiquetas de rango a nombres más descriptivos
        Map<String, Map<String, String>> labelTranslations = new HashMap<>();
        
        // Traducciones para reviews_range
        Map<String, String> reviewsLabels = new HashMap<>();
        reviewsLabels.put("0", "Sin reseñas");
        reviewsLabels.put("1-5", "Pocas reseñas");
        reviewsLabels.put("6-34", "Algunas reseñas");
        reviewsLabels.put("35-110", "Muchas reseñas");
        reviewsLabels.put("111+", "Muchísimas reseñas");
        labelTranslations.put("reviews_range", reviewsLabels);
        
        // Traducciones para price_range
        Map<String, String> priceLabels = new HashMap<>();
        priceLabels.put("barato", "Barato");
        priceLabels.put("asequible", "Asequible");
        priceLabels.put("caro", "Caro");
        labelTranslations.put("price_range", priceLabels);
        
        // Traducciones para rating_range
        Map<String, String> ratingLabels = new HashMap<>();
        ratingLabels.put("0-2", "1-2 estrellas");
        ratingLabels.put("2-3", "2-3 estrellas");
        ratingLabels.put("3-4", "3-4 estrellas");
        ratingLabels.put("4-4.5", "4 estrellas");
        ratingLabels.put("4.5-5", "4.5-5 estrellas");
        labelTranslations.put("rating_range", ratingLabels);

        // Manejar facetas jerárquicas de property_type de forma especial
        // Mostrar primero las categorías principales, y al seleccionar una, mostrar sus subfacetas
        if (facetsData.containsKey("property_type")) {
            List<LabelAndValue> propertyTypeValues = facetsData.get("property_type");
            Map<String, List<LabelAndValue>> categoryGroups = new LinkedHashMap<>();
            Map<String, Long> categoryCounts = new HashMap<>();
            
            // Agrupar por categoría principal (primer nivel del path jerárquico)
            for (LabelAndValue lv : propertyTypeValues) {
                String path = lv.label;
                String[] parts = path.split("/", 2);
                if (parts.length == 2) {
                    String category = parts[0];
                    String specificType = parts[1];
                    categoryGroups.computeIfAbsent(category, k -> new ArrayList<>())
                        .add(new LabelAndValue(specificType, lv.value));
                    // Sumar el conteo de la categoría principal
                    categoryCounts.put(category, categoryCounts.getOrDefault(category, 0L) + lv.value.longValue());
                } else {
                    // Si no es jerárquico, poner en "other"
                    categoryGroups.computeIfAbsent("other", k -> new ArrayList<>())
                        .add(lv);
                    categoryCounts.put("other", categoryCounts.getOrDefault("other", 0L) + lv.value.longValue());
                }
            }
            
            // Crear un TitledPane principal para "Tipo de Propiedad"
            VBox mainPropertyTypeBox = new VBox(6);
            
            // Determinar qué categorías tienen facetas activas para expandirlas
            // Incluye tanto categorías seleccionadas directamente como aquellas con subfacetas activas
            Map<String, Boolean> categoriesWithActiveFacets = new HashMap<>();
            if (activeFacets != null && activeFacets.containsKey("property_type")) {
                for (String activePath : activeFacets.get("property_type")) {
                    if (activePath.contains("/")) {
                        // Subfaceta específica: "home/entire home"
                        String[] parts = activePath.split("/", 2);
                        if (parts.length == 2) {
                            categoriesWithActiveFacets.put(parts[0], true);
                        }
                    } else {
                        // Categoría principal seleccionada directamente: "home"
                        categoriesWithActiveFacets.put(activePath, true);
                    }
                }
            }
            
            for (Map.Entry<String, List<LabelAndValue>> categoryEntry : categoryGroups.entrySet()) {
                String category = categoryEntry.getKey();
                List<LabelAndValue> subTypes = categoryEntry.getValue();
                
                // Checkbox para la categoría principal
                long categoryCount = categoryCounts.getOrDefault(category, 0L);
                String categoryDisplayName = propertyTypeCategoryNames.getOrDefault(category, category);
                CheckBox categoryCheckBox = new CheckBox(categoryDisplayName + " (" + categoryCount + ")");
                
                // Determinar si la categoría está seleccionada directamente (sin subfaceta específica)
                boolean categorySelectedDirectly = activeFacets != null && 
                    activeFacets.containsKey("property_type") && 
                    activeFacets.get("property_type").contains(category);
                
                // Determinar si hay subfacetas activas para esta categoría
                boolean hasActiveSubFacets = categoriesWithActiveFacets.containsKey(category);
                
                // La categoría está "seleccionada" si está seleccionada directamente O tiene subfacetas activas
                boolean categorySelected = categorySelectedDirectly || hasActiveSubFacets;
                categoryCheckBox.setSelected(categorySelectedDirectly); // Solo marcar si está seleccionada directamente
                
                // Asociar el checkbox de categoría con su path jerárquico (solo categoría, sin subfaceta)
                // Esto permite filtrar por toda la categoría usando la jerarquía de Lucene
                categoryCheckBox.setUserData(new FacetSelection("property_type", category));
                
                // Contenedor para las subfacetas de esta categoría
                VBox subFacetsBox = new VBox(4);
                subFacetsBox.setPadding(new Insets(5, 0, 0, 20)); // Indentación para subfacetas
                
                // Extraer subfacetas activas para esta categoría
                List<String> activeForCategory = activeFacets != null ? 
                    extractActiveForCategory(activeFacets.get("property_type"), category) : null;
                
                for (LabelAndValue lv : subTypes) {
                    String label = lv.label;
                    long count = lv.value.longValue();
                    String fullPath = category + "/" + label;
                    
                    CheckBox cb = new CheckBox(label + " (" + count + ")");
                    cb.setUserData(new FacetSelection("property_type", fullPath));
                    // Marcar como seleccionado si está en las facetas activas
                    // Esto incluye tanto cuando se selecciona directamente como cuando viene de la búsqueda
                    if (activeForCategory != null && activeForCategory.contains(fullPath)) {
                        cb.setSelected(true);
                    }
                    subFacetsBox.getChildren().add(cb);
                }
                
                // Hacer que el checkbox de categoría controle la visibilidad de las subfacetas
                // Mostrar subfacetas si la categoría está seleccionada directamente O si hay subfacetas activas
                subFacetsBox.setVisible(categorySelected);
                subFacetsBox.setManaged(categorySelected);
                
                categoryCheckBox.setOnAction(e -> {
                    boolean isSelected = categoryCheckBox.isSelected();
                    subFacetsBox.setVisible(isSelected);
                    subFacetsBox.setManaged(isSelected);
                    // Cuando se selecciona la categoría principal, NO seleccionar automáticamente las subfacetas
                    // Esto permite usar la jerarquía de Lucene: seleccionar "home" filtra por TODOS sus hijos
                    // Si el usuario quiere ser más específico, puede deseleccionar la categoría y seleccionar subfacetas individuales
                    // Cuando se deselecciona la categoría, sí deseleccionar todas las subfacetas
                    if (!isSelected) {
                        for (javafx.scene.Node node : subFacetsBox.getChildren()) {
                            if (node instanceof CheckBox) {
                                ((CheckBox) node).setSelected(false);
                            }
                        }
                    }
                });
                
                // Contenedor para la categoría y sus subfacetas
                VBox categoryContainer = new VBox(4);
                categoryContainer.getChildren().addAll(categoryCheckBox, subFacetsBox);
                
                mainPropertyTypeBox.getChildren().add(categoryContainer);
            }
            
            TitledPane mainPane = new TitledPane("Filtrar por Tipo de Propiedad", mainPropertyTypeBox);
            mainPane.setExpanded(true); // Expandido por defecto para ver las categorías
            facetsContainer.getChildren().add(mainPane);
            
            // Remover property_type de facetsData para no procesarlo dos veces
            Map<String, List<LabelAndValue>> otherFacets = new LinkedHashMap<>(facetsData);
            otherFacets.remove("property_type");
            facetsData = otherFacets;
        }
        
        // Procesar las demás facetas normalmente
        for (Map.Entry<String, List<LabelAndValue>> entry : facetsData.entrySet()) {
            String dim = entry.getKey();
            // Omitir property_type_simple ya que usamos la versión jerárquica property_type
            if ("property_type_simple".equals(dim)) {
                continue;
            }
            List<LabelAndValue> values = entry.getValue();
            VBox box = new VBox(4);

            List<String> activeForDim = activeFacets != null ? activeFacets.get(dim) : null;

            for (LabelAndValue lv : values) {
                String label = lv.label;
                long count = lv.value.longValue();
                
                // Traducir la etiqueta si hay traducción disponible
                Map<String, String> translations = labelTranslations.get(dim);
                String displayLabel = (translations != null && translations.containsKey(label)) 
                    ? translations.get(label) 
                    : label;
                
                CheckBox cb = new CheckBox(displayLabel + " (" + count + ")");
                cb.setUserData(new FacetSelection(dim, label));
                if (activeForDim != null && activeForDim.contains(label)) {
                    cb.setSelected(true);
                }
                box.getChildren().add(cb);
            }

            // Usar nombre amigable si está disponible, sino usar el nombre técnico
            String displayDim = friendlyNames.getOrDefault(dim, dim);
            TitledPane pane = new TitledPane("Filtrar por " + displayDim, box);
            pane.setExpanded(false);
            facetsContainer.getChildren().add(pane);
        }
    }

    /**
     * Aplica las facetas marcadas en el sidebar (botón "Aplicar filtros").
     * Maneja tanto facetas simples como jerárquicas anidadas.
     */
    private void applyFacetSelection() {
        Map<String, List<String>> selectedFacets = new LinkedHashMap<>();
        if (facetsContainer != null) {
            for (javafx.scene.Node node : facetsContainer.getChildren()) {
                if (node instanceof TitledPane) {
                    TitledPane pane = (TitledPane) node;
                    javafx.scene.Node content = pane.getContent();
                    if (content instanceof VBox) {
                        VBox vbox = (VBox) content;
                        // Recursivamente buscar checkboxes en VBoxes anidados (para facetas jerárquicas)
                        collectCheckboxesFromVBox(vbox, selectedFacets);
                    }
                }
            }
        }

        final String queryText = simpleQueryField.getText() != null
                ? simpleQueryField.getText().trim()
                : "";
        final String neighbourhood = neighbourhoodField.getText() != null
                ? neighbourhoodField.getText().trim().toLowerCase()
                : "";
        final String minPriceText = minPriceField.getText() != null ? minPriceField.getText().trim() : "";
        final String maxPriceText = maxPriceField.getText() != null ? maxPriceField.getText().trim() : "";

        statusLabel.setText("Aplicando filtros por facetas...");
        runSearchInBackground(queryText, neighbourhood, minPriceText, maxPriceText, selectedFacets);
    }

    /**
     * Limpia las facetas seleccionadas (botón "Quitar filtros") y relanza la
     * búsqueda solo con los campos de texto.
     * Maneja estructuras anidadas recursivamente.
     */
    private void clearFacetSelection() {
        if (facetsContainer != null) {
            for (javafx.scene.Node node : facetsContainer.getChildren()) {
                if (node instanceof TitledPane) {
                    TitledPane pane = (TitledPane) node;
                    javafx.scene.Node content = pane.getContent();
                    if (content instanceof VBox) {
                        clearCheckboxesFromVBox((VBox) content);
                    }
                }
            }
        }

        final String queryText = simpleQueryField.getText() != null
                ? simpleQueryField.getText().trim()
                : "";
        final String neighbourhood = neighbourhoodField.getText() != null
                ? neighbourhoodField.getText().trim().toLowerCase()
                : "";
        final String minPriceText = minPriceField.getText() != null ? minPriceField.getText().trim() : "";
        final String maxPriceText = maxPriceField.getText() != null ? maxPriceField.getText().trim() : "";

        statusLabel.setText("Filtros de facetas limpiados.");
        runSearchInBackground(queryText, neighbourhood, minPriceText, maxPriceText, null);
    }

    /**
     * Helper para extraer facetas activas de una categoría específica.
     */
    private List<String> extractActiveForCategory(List<String> activePaths, String category) {
        if (activePaths == null) {
            return null;
        }
        List<String> result = new ArrayList<>();
        for (String path : activePaths) {
            if (path.startsWith(category + "/")) {
                result.add(path);
            }
        }
        return result.isEmpty() ? null : result;
    }

    /**
     * Recursivamente recoge checkboxes de un VBox, manejando estructuras anidadas
     * (para facetas jerárquicas con TitledPanes dentro de TitledPanes).
     * Para property_type jerárquico, maneja correctamente la selección de categorías principales
     * vs subfacetas específicas.
     */
    private void collectCheckboxesFromVBox(VBox vbox, Map<String, List<String>> selectedFacets) {
        // Recoger todas las selecciones directamente sin optimización previa
        // La optimización se hará en la construcción del DrillDownQuery
        for (javafx.scene.Node child : vbox.getChildren()) {
            if (child instanceof CheckBox) {
                CheckBox cb = (CheckBox) child;
                Object ud = cb.getUserData();
                if (cb.isSelected() && ud instanceof FacetSelection) {
                    FacetSelection fs = (FacetSelection) ud;
                    selectedFacets
                            .computeIfAbsent(fs.dimension, k -> new ArrayList<>())
                            .add(fs.label);
                }
            } else if (child instanceof TitledPane) {
                // Manejar TitledPanes anidados (categorías dentro de property_type)
                TitledPane nestedPane = (TitledPane) child;
                javafx.scene.Node nestedContent = nestedPane.getContent();
                if (nestedContent instanceof VBox) {
                    collectCheckboxesFromVBox((VBox) nestedContent, selectedFacets);
                }
            } else if (child instanceof VBox) {
                // También manejar VBoxes anidados directamente (para categoryContainer)
                collectCheckboxesFromVBox((VBox) child, selectedFacets);
            }
        }
    }

    /**
     * Recursivamente limpia checkboxes de un VBox, manejando estructuras anidadas.
     */
    private void clearCheckboxesFromVBox(VBox vbox) {
        for (javafx.scene.Node child : vbox.getChildren()) {
            if (child instanceof CheckBox) {
                ((CheckBox) child).setSelected(false);
            } else if (child instanceof TitledPane) {
                TitledPane nestedPane = (TitledPane) child;
                javafx.scene.Node nestedContent = nestedPane.getContent();
                if (nestedContent instanceof VBox) {
                    clearCheckboxesFromVBox((VBox) nestedContent);
                }
            } else if (child instanceof VBox) {
                clearCheckboxesFromVBox((VBox) child);
            }
        }
    }

    /**
     * Pequeña estructura para asociar cada CheckBox con su dimensión/etiqueta real.
     */
    private static class FacetSelection {
        final String dimension;
        final String label;

        FacetSelection(String dimension, String label) {
            this.dimension = dimension;
            this.label = label;
        }
    }
}