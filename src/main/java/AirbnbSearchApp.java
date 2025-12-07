import javafx.application.Application;
import javafx.application.HostServices;
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
import javafx.scene.text.Text;
import javafx.scene.text.TextFlow;
import static javafx.scene.layout.Region.USE_COMPUTED_SIZE;
import javafx.stage.Stage;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.DoublePoint;
import org.apache.lucene.document.IntPoint;
import org.apache.lucene.document.LatLonPoint;
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
import org.apache.lucene.search.highlight.Highlighter;
import org.apache.lucene.search.highlight.QueryScorer;
import org.apache.lucene.search.highlight.SimpleHTMLFormatter;
import org.apache.lucene.search.highlight.SimpleSpanFragmenter;
import org.apache.lucene.analysis.TokenStream;

import java.io.StringReader;
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
        private final SimpleDoubleProperty rating = new SimpleDoubleProperty();
        private final SimpleIntegerProperty reviews = new SimpleIntegerProperty();
        private final SimpleIntegerProperty bedrooms = new SimpleIntegerProperty();
        private final SimpleIntegerProperty bathrooms = new SimpleIntegerProperty();
        private final SimpleStringProperty listingUrl = new SimpleStringProperty();
        private final SimpleStringProperty amenities = new SimpleStringProperty();
        private final SimpleStringProperty description = new SimpleStringProperty();
        private final SimpleStringProperty descriptionHighlighted = new SimpleStringProperty();

        public PropertyResult(int id, String name, String neighbourhood, String propertyType, Double price,
                             Double rating, Integer reviews, Integer bedrooms, Integer bathrooms, String listingUrl,
                             String amenities, String description, String descriptionHighlighted) {
            this.id.set(id);
            this.name.set(name != null ? name : "");
            this.neighbourhood.set(neighbourhood != null ? neighbourhood : "");
            this.propertyType.set(propertyType != null ? propertyType : "");
            this.price.set(price != null ? price : 0.0);
            this.rating.set(rating != null ? rating : 0.0);
            this.reviews.set(reviews != null ? reviews : 0);
            this.bedrooms.set(bedrooms != null ? bedrooms : 0);
            this.bathrooms.set(bathrooms != null ? bathrooms : 0);
            this.listingUrl.set(listingUrl != null ? listingUrl : "");
            this.amenities.set(amenities != null ? amenities : "");
            this.description.set(description != null ? description : "");
            this.descriptionHighlighted.set(descriptionHighlighted != null ? descriptionHighlighted : description != null ? description : "");
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

        public double getRating() {
            return rating.get();
        }

        public int getReviews() {
            return reviews.get();
        }

        public int getBedrooms() {
            return bedrooms.get();
        }

        public int getBathrooms() {
            return bathrooms.get();
        }

        public String getListingUrl() {
            return listingUrl.get();
        }

        public String getAmenities() {
            return amenities.get();
        }

        public String getDescription() {
            return description.get();
        }

        public String getDescriptionHighlighted() {
            return descriptionHighlighted.get();
        }
    }

    // Parámetros básicos: se podría exponer como argumento de línea de comandos
    private String indexRoot = "./index_root";
    private static final int MAX_RESULTS = 1000; // Límite máximo de resultados a recuperar

    // ========== UI Layout Constants ==========
    private static final double ROOT_PADDING = 10.0;
    private static final double HEADER_PADDING_BOTTOM = 10.0;
    private static final double HEADER_SPACING = 8.0;
    private static final double ADVANCED_SCROLL_MAX_HEIGHT = 400.0;
    private static final double ADVANCED_CONTENT_SPACING = 10.0;
    private static final double ADVANCED_CONTENT_PADDING = 10.0;
    private static final double FIELD_GROUP_SPACING = 8.0;
    private static final double SIDEBAR_SPACING = 8.0;
    private static final double SIDEBAR_PADDING_RIGHT = 10.0;
    private static final double SIDEBAR_PREF_WIDTH = 280.0;
    private static final double FACETS_CONTAINER_SPACING = 6.0;
    private static final double FACETS_CONTAINER_PADDING_TOP = 6.0;
    private static final double STATUS_BAR_SPACING = 10.0;
    private static final double STATUS_BAR_PADDING_TOP = 8.0;
    private static final double SUBFACETS_BOX_SPACING = 4.0;
    private static final double SUBFACETS_BOX_PADDING_TOP = 5.0;
    private static final double SUBFACETS_BOX_PADDING_LEFT = 20.0;
    private static final double CATEGORY_CONTAINER_SPACING = 4.0;
    private static final double FACET_BOX_SPACING = 4.0;
    private static final double PROPERTY_TYPE_BOX_SPACING = 6.0;

    // ========== Scene Dimensions ==========
    private static final double SCENE_WIDTH = 1100.0;
    private static final double SCENE_HEIGHT = 700.0;

    // ========== Column Width Constraints ==========
    private static final double COL_NAME_MIN_WIDTH = 80.0;
    private static final double COL_NAME_MAX_WIDTH = 400.0;
    private static final double COL_DESCRIPTION_MIN_WIDTH = 200.0;
    private static final double COL_DESCRIPTION_MAX_WIDTH = 600.0;
    private static final double COL_DESCRIPTION_HEIGHT = 200.0;
    private static final double COL_DESCRIPTION_SCROLLPANE_BORDER_SUBTRACT = 5.0;
    private static final double COL_DESCRIPTION_TEXTFLOW_BORDER_SUBTRACT = 10.0;
    private static final double COL_NEIGHBOURHOOD_MIN_WIDTH = 60.0;
    private static final double COL_NEIGHBOURHOOD_MAX_WIDTH = 200.0;
    private static final double COL_TYPE_MIN_WIDTH = 50.0;
    private static final double COL_TYPE_MAX_WIDTH = 200.0;
    private static final double COL_PRICE_MIN_WIDTH = 50.0;
    private static final double COL_PRICE_MAX_WIDTH = 100.0;
    private static final double COL_RATING_MIN_WIDTH = 50.0;
    private static final double COL_RATING_MAX_WIDTH = 80.0;
    private static final double COL_REVIEWS_MIN_WIDTH = 50.0;
    private static final double COL_REVIEWS_MAX_WIDTH = 100.0;
    private static final double COL_BEDROOMS_MIN_WIDTH = 50.0;
    private static final double COL_BEDROOMS_MAX_WIDTH = 120.0;
    private static final double COL_BATHROOMS_MIN_WIDTH = 40.0;
    private static final double COL_BATHROOMS_MAX_WIDTH = 80.0;
    private static final double COL_URL_MIN_WIDTH = 40.0;
    private static final double COL_URL_MAX_WIDTH = 80.0;
    private static final double COL_AMENITIES_MIN_WIDTH = 80.0;
    private static final double COL_AMENITIES_MAX_WIDTH = 500.0;

    // ========== Field Widths ==========
    private static final double FIELD_PRICE_PREF_WIDTH = 100.0;

    // ========== Font Sizes ==========
    private static final double FONT_SIZE_HELP_TEXT = 10.0;
    private static final double FONT_SIZE_QUERY_LABEL = 11.0;

    // ========== Facet Query Limits ==========
    private static final int FACET_DIMS_INITIAL_LIMIT = 20;
    private static final int FACET_DIMS_ALL_LIMIT = 1000;

    // ========== Numeric Query Constants ==========
    private static final double DOUBLE_EPSILON = 0.01;
    private static final int INT_EPSILON = 1;
    private static final double DOUBLE_DEFAULT_MIN = 0.0;

    // ========== HTML Tag Parsing Constants ==========
    private static final int HTML_TAG_MARK_OPEN_LENGTH = 6; // "<mark>"
    private static final int HTML_TAG_MARK_CLOSE_LENGTH = 7; // "</mark>"
    private static final String HTML_TAG_MARK_OPEN = "<mark>";
    private static final String HTML_TAG_MARK_CLOSE = "</mark>";

    // ========== Numeric Format Constants ==========
    private static final String FORMAT_PRICE = "%.2f";
    private static final String FORMAT_RATING = "%.1f";

    // Componentes UI principales
    private TextField simpleQueryField;
    private TextField neighbourhoodField;
    private TextField minPriceField;
    private TextField maxPriceField;
    // Campos adicionales para búsqueda avanzada (inspirados en BusquedasLucene)
    private TextField ratingField; // review_scores_rating con operadores (>=, >, <, <=, =)
    private TextField reviewsField; // number_of_reviews con operadores
    private TextField bedroomsField; // bedrooms con operadores
    private TextField bathroomsField; // bathrooms con operadores
    private TextField amenityField; // amenity (búsqueda textual)
    private TextField propertyTypeField; // property_type (búsqueda textual)
    // Búsqueda geográfica
    private TextField latField;
    private TextField lonField;
    private TextField radiusField;
    private Label statusLabel;
    // Contenedor para grupos de facetas (similar a la columna de filtros de un buscador web)
    private VBox facetsContainer;
    private TableView<PropertyResult> resultsTable;
    private TableColumn<PropertyResult, String> descriptionCol;
    private final ObservableList<PropertyResult> resultsData = FXCollections.observableArrayList();
    private HostServices hostServices;

    @Override
    public void start(Stage primaryStage) {
        // Guardar referencia a HostServices para abrir URLs
        hostServices = getHostServices();
        
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
        root.setPadding(new Insets(ROOT_PADDING));

        root.setTop(createSearchHeader());
        root.setLeft(createFacetSidebar());
        root.setCenter(createResultsTable());
        root.setBottom(createStatusBar());

        Scene scene = new Scene(root, SCENE_WIDTH, SCENE_HEIGHT);
        primaryStage.setTitle("Airbnb Lucene Search (JavaFX Prototype)");
        primaryStage.setScene(scene);
        primaryStage.show();
    }

    /**
     * Crea la parte superior: búsqueda simple + búsqueda avanzada plegable.
     */
    private VBox createSearchHeader() {
        VBox header = new VBox(HEADER_SPACING);
        header.setPadding(new Insets(0, 0, HEADER_PADDING_BOTTOM, 0));

        // Fila de búsqueda simple
        HBox simpleRow = new HBox(HEADER_SPACING);
        simpleRow.setAlignment(Pos.CENTER_LEFT);

        simpleQueryField = new TextField();
        simpleQueryField.setPromptText("Buscar en mega campo (ej: 'pool AND 3 bedrooms')");
        simpleQueryField.setOnAction(e -> executeSearch()); // Enter lanza búsqueda

        Button searchButton = new Button("Buscar");
        searchButton.setDefaultButton(true);
        searchButton.setOnAction(e -> executeSearch());

        Button clearButton = new Button("Reset");
        clearButton.setOnAction(e -> clearFiltersAndResults());

        CheckBox hideDescriptionCheck = new CheckBox("Ocultar Descripción");
        hideDescriptionCheck.setSelected(false);
        hideDescriptionCheck.setOnAction(e -> {
            if (descriptionCol != null) {
                descriptionCol.setVisible(!hideDescriptionCheck.isSelected());
            }
        });

        simpleRow.getChildren().addAll(new Label("Consulta:"), simpleQueryField, searchButton, clearButton, hideDescriptionCheck);
        HBox.setHgrow(simpleQueryField, Priority.ALWAYS);

        // Sección de búsqueda avanzada (inspirada en BusquedasLucene)
        ScrollPane advancedScroll = new ScrollPane();
        advancedScroll.setFitToWidth(true);
        advancedScroll.setHbarPolicy(ScrollPane.ScrollBarPolicy.NEVER);
        advancedScroll.setVbarPolicy(ScrollPane.ScrollBarPolicy.AS_NEEDED);
        advancedScroll.setMaxHeight(ADVANCED_SCROLL_MAX_HEIGHT);
        
        VBox advancedContent = new VBox(ADVANCED_CONTENT_SPACING);
        advancedContent.setPadding(new Insets(ADVANCED_CONTENT_PADDING));
        
        // Grupo 1: Campos de texto y categorías
        TitledPane textGroup = new TitledPane("Campos de Texto y Categorías", createTextFieldsGroup());
        textGroup.setExpanded(true);
        
        // Grupo 2: Campos numéricos con operadores
        TitledPane numericGroup = new TitledPane("Campos Numéricos (con operadores: >=, >, <, <=, =)", createNumericFieldsGroup());
        numericGroup.setExpanded(true);
        
        // Grupo 3: Búsqueda geográfica
        TitledPane geoGroup = new TitledPane("Búsqueda Geográfica", createGeographicFieldsGroup());
        geoGroup.setExpanded(false);
        
        advancedContent.getChildren().addAll(textGroup, numericGroup, geoGroup);
        advancedScroll.setContent(advancedContent);

        TitledPane advancedPane = new TitledPane("Búsqueda avanzada", advancedScroll);
        advancedPane.setExpanded(false);

        header.getChildren().addAll(simpleRow, advancedPane);
        return header;
    }

    /**
     * Crea el grupo de campos de texto y categorías para la búsqueda avanzada.
     */
    private VBox createTextFieldsGroup() {
        VBox group = new VBox(FIELD_GROUP_SPACING);
        
        // Barrio
        HBox neighbourhoodRow = new HBox(FIELD_GROUP_SPACING);
        neighbourhoodRow.setAlignment(Pos.CENTER_LEFT);
        neighbourhoodField = new TextField();
        neighbourhoodField.setPromptText("neighbourhood_cleansed (ej: hollywood)");
        neighbourhoodRow.getChildren().addAll(new Label("Barrio:"), neighbourhoodField);
        HBox.setHgrow(neighbourhoodField, Priority.ALWAYS);
        
        // Amenidad
        HBox amenityRow = new HBox(FIELD_GROUP_SPACING);
        amenityRow.setAlignment(Pos.CENTER_LEFT);
        amenityField = new TextField();
        amenityField.setPromptText("amenity (ej: pool, wifi, parking)");
        amenityRow.getChildren().addAll(new Label("Amenidad:"), amenityField);
        HBox.setHgrow(amenityField, Priority.ALWAYS);
        
        // Tipo de propiedad
        HBox propertyTypeRow = new HBox(FIELD_GROUP_SPACING);
        propertyTypeRow.setAlignment(Pos.CENTER_LEFT);
        propertyTypeField = new TextField();
        propertyTypeField.setPromptText("property_type (ej: entire home, private room)");
        propertyTypeRow.getChildren().addAll(new Label("Tipo:"), propertyTypeField);
        HBox.setHgrow(propertyTypeField, Priority.ALWAYS);
        
        group.getChildren().addAll(neighbourhoodRow, amenityRow, propertyTypeRow);
        return group;
    }

    /**
     * Crea el grupo de campos numéricos con operadores para la búsqueda avanzada.
     */
    private VBox createNumericFieldsGroup() {
        VBox group = new VBox(FIELD_GROUP_SPACING);
        
        // Precio (rango)
        HBox priceRow = new HBox(FIELD_GROUP_SPACING);
        priceRow.setAlignment(Pos.CENTER_LEFT);
        minPriceField = new TextField();
        minPriceField.setPromptText("Precio mín.");
        minPriceField.setPrefWidth(FIELD_PRICE_PREF_WIDTH);
        maxPriceField = new TextField();
        maxPriceField.setPromptText("Precio máx.");
        maxPriceField.setPrefWidth(FIELD_PRICE_PREF_WIDTH);
        Label priceHelp = new Label("(o use operadores: >=100, >100, <200, <=200, =150)");
        priceHelp.setStyle("-fx-font-size: " + FONT_SIZE_HELP_TEXT + "px; -fx-text-fill: #666;");
        priceRow.getChildren().addAll(new Label("Precio:"), minPriceField, new Label("-"), maxPriceField, priceHelp);
        
        // Rating con operadores
        HBox ratingRow = new HBox(FIELD_GROUP_SPACING);
        ratingRow.setAlignment(Pos.CENTER_LEFT);
        ratingField = new TextField();
        ratingField.setPromptText(">=4.7, >4.5, <3.0, =5.0");
        Label ratingHelp = new Label("(ej: >=4.7 para rating >= 4.7)");
        ratingHelp.setStyle("-fx-font-size: " + FONT_SIZE_HELP_TEXT + "px; -fx-text-fill: #666;");
        ratingRow.getChildren().addAll(new Label("Rating:"), ratingField, ratingHelp);
        HBox.setHgrow(ratingField, Priority.ALWAYS);
        
        // Número de reseñas con operadores
        HBox reviewsRow = new HBox(FIELD_GROUP_SPACING);
        reviewsRow.setAlignment(Pos.CENTER_LEFT);
        reviewsField = new TextField();
        reviewsField.setPromptText(">=10, >0, <5, =0");
        Label reviewsHelp = new Label("(ej: >=10 para 10+ reseñas)");
        reviewsHelp.setStyle("-fx-font-size: " + FONT_SIZE_HELP_TEXT + "px; -fx-text-fill: #666;");
        reviewsRow.getChildren().addAll(new Label("Reseñas:"), reviewsField, reviewsHelp);
        HBox.setHgrow(reviewsField, Priority.ALWAYS);
        
        // Habitaciones con operadores
        HBox bedroomsRow = new HBox(FIELD_GROUP_SPACING);
        bedroomsRow.setAlignment(Pos.CENTER_LEFT);
        bedroomsField = new TextField();
        bedroomsField.setPromptText(">=2, >1, <5, =3");
        Label bedroomsHelp = new Label("(ej: >=2 para 2+ habitaciones)");
        bedroomsHelp.setStyle("-fx-font-size: " + FONT_SIZE_HELP_TEXT + "px; -fx-text-fill: #666;");
        bedroomsRow.getChildren().addAll(new Label("Habitaciones:"), bedroomsField, bedroomsHelp);
        HBox.setHgrow(bedroomsField, Priority.ALWAYS);
        
        // Baños con operadores
        HBox bathroomsRow = new HBox(FIELD_GROUP_SPACING);
        bathroomsRow.setAlignment(Pos.CENTER_LEFT);
        bathroomsField = new TextField();
        bathroomsField.setPromptText(">=1, >0, <3, =2");
        Label bathroomsHelp = new Label("(ej: >=1 para 1+ baños)");
        bathroomsHelp.setStyle("-fx-font-size: " + FONT_SIZE_HELP_TEXT + "px; -fx-text-fill: #666;");
        bathroomsRow.getChildren().addAll(new Label("Baños:"), bathroomsField, bathroomsHelp);
        HBox.setHgrow(bathroomsField, Priority.ALWAYS);
        
        group.getChildren().addAll(priceRow, ratingRow, reviewsRow, bedroomsRow, bathroomsRow);
        return group;
    }

    /**
     * Crea el grupo de campos geográficos para la búsqueda avanzada.
     */
    private VBox createGeographicFieldsGroup() {
        VBox group = new VBox(FIELD_GROUP_SPACING);
        
        HBox latRow = new HBox(FIELD_GROUP_SPACING);
        latRow.setAlignment(Pos.CENTER_LEFT);
        latField = new TextField();
        latField.setPromptText("34.0522 (ej: Los Angeles)");
        Label latHelp = new Label("(rango típico: 33.339 - 34.811)");
        latHelp.setStyle("-fx-font-size: " + FONT_SIZE_HELP_TEXT + "px; -fx-text-fill: #666;");
        latRow.getChildren().addAll(new Label("Latitud:"), latField, latHelp);
        HBox.setHgrow(latField, Priority.ALWAYS);
        
        HBox lonRow = new HBox(FIELD_GROUP_SPACING);
        lonRow.setAlignment(Pos.CENTER_LEFT);
        lonField = new TextField();
        lonField.setPromptText("-118.2437 (ej: Los Angeles)");
        Label lonHelp = new Label("(rango típico: -118.917 - -117.654)");
        lonHelp.setStyle("-fx-font-size: " + FONT_SIZE_HELP_TEXT + "px; -fx-text-fill: #666;");
        lonRow.getChildren().addAll(new Label("Longitud:"), lonField, lonHelp);
        HBox.setHgrow(lonField, Priority.ALWAYS);
        
        HBox radiusRow = new HBox(FIELD_GROUP_SPACING);
        radiusRow.setAlignment(Pos.CENTER_LEFT);
        radiusField = new TextField();
        radiusField.setPromptText("5000 (metros)");
        Label radiusHelp = new Label("(ej: 5000 = 5km, 1000 = 1km)");
        radiusHelp.setStyle("-fx-font-size: " + FONT_SIZE_HELP_TEXT + "px; -fx-text-fill: #666;");
        radiusRow.getChildren().addAll(new Label("Radio:"), radiusField, radiusHelp);
        HBox.setHgrow(radiusField, Priority.ALWAYS);
        
        group.getChildren().addAll(latRow, lonRow, radiusRow);
        return group;
    }

    /**
     * Crea un sidebar para facetas con estructura similar a la de un buscador web:
     * botones de "Aplicar/Quitar filtros" y bloques plegables por categoría.
     */
    private VBox createFacetSidebar() {
        VBox side = new VBox(SIDEBAR_SPACING);
        side.setPadding(new Insets(0, SIDEBAR_PADDING_RIGHT, 0, 0));
        side.setPrefWidth(SIDEBAR_PREF_WIDTH);

        // Fila de botones estilo "Aplicar filtros" / "Quitar filtros"
        HBox buttonsRow = new HBox(SIDEBAR_SPACING);
        buttonsRow.setAlignment(Pos.CENTER);
        Button applyButton = new Button("Aplicar filtros");
        applyButton.setOnAction(e -> applyFacetSelection());
        Button clearButton = new Button("Quitar filtros");
        clearButton.setOnAction(e -> clearFacetSelection());
        buttonsRow.getChildren().addAll(applyButton, clearButton);

        Label title = new Label("Filtros por facetas");
        title.setStyle("-fx-font-weight: bold;");

        facetsContainer = new VBox(FACETS_CONTAINER_SPACING);
        facetsContainer.setPadding(new Insets(FACETS_CONTAINER_PADDING_TOP, 0, 0, 0));
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

        TableColumn<PropertyResult, String> nameCol = new TableColumn<>("Nombre");
        nameCol.setCellValueFactory(new PropertyValueFactory<>("name"));
        nameCol.setPrefWidth(USE_COMPUTED_SIZE);
        nameCol.setMinWidth(COL_NAME_MIN_WIDTH); // Minimum width for header
        nameCol.setMaxWidth(COL_NAME_MAX_WIDTH); // Maximum width to prevent excessive expansion
        nameCol.setResizable(true);

        descriptionCol = new TableColumn<>("Descripción");
        descriptionCol.setCellValueFactory(new PropertyValueFactory<>("descriptionHighlighted"));
        descriptionCol.setPrefWidth(USE_COMPUTED_SIZE);
        descriptionCol.setMinWidth(COL_DESCRIPTION_MIN_WIDTH);
        descriptionCol.setMaxWidth(COL_DESCRIPTION_MAX_WIDTH);
        descriptionCol.setResizable(true);
        descriptionCol.setVisible(true);
        descriptionCol.setCellFactory(column -> new TableCell<PropertyResult, String>() {
            @Override
            protected void updateItem(String highlightedText, boolean empty) {
                super.updateItem(highlightedText, empty);
                if (empty || highlightedText == null) {
                    setGraphic(null);
                    setPrefHeight(USE_COMPUTED_SIZE);
                } else {
                    // Parsear el texto con etiquetas <mark> y crear TextFlow
                    TextFlow textFlow = parseHighlightedText(highlightedText);
                    
                    // Usar ScrollPane para permitir scroll del contenido sin cortar texto
                    // Esto previene que el TextFlow haga crecer la fila indefinidamente
                    ScrollPane scrollPane = new ScrollPane(textFlow);
                    // Bind width to column width for flexibility
                    scrollPane.prefWidthProperty().bind(column.widthProperty().subtract(COL_DESCRIPTION_SCROLLPANE_BORDER_SUBTRACT));
                    scrollPane.maxWidthProperty().bind(column.widthProperty().subtract(COL_DESCRIPTION_SCROLLPANE_BORDER_SUBTRACT));
                    scrollPane.setMaxHeight(COL_DESCRIPTION_HEIGHT); 
                    scrollPane.setPrefHeight(COL_DESCRIPTION_HEIGHT);
                    scrollPane.setFitToWidth(true);
                    scrollPane.setFitToHeight(false);
                    scrollPane.setHbarPolicy(ScrollPane.ScrollBarPolicy.NEVER);
                    scrollPane.setVbarPolicy(ScrollPane.ScrollBarPolicy.AS_NEEDED);
                    scrollPane.setStyle("-fx-background-color: transparent; -fx-border-color: transparent;");
                    scrollPane.setPadding(Insets.EMPTY);
                    
                    // Bind TextFlow width to ScrollPane width
                    textFlow.maxWidthProperty().bind(scrollPane.widthProperty().subtract(COL_DESCRIPTION_TEXTFLOW_BORDER_SUBTRACT));
                    textFlow.prefWidthProperty().bind(scrollPane.widthProperty().subtract(COL_DESCRIPTION_TEXTFLOW_BORDER_SUBTRACT));
                    
                    setGraphic(scrollPane);
                    // Fijar altura de la celda para evitar crecimiento
                    setPrefHeight(COL_DESCRIPTION_HEIGHT);
                    setMaxHeight(COL_DESCRIPTION_HEIGHT);
                }
            }
        });

        TableColumn<PropertyResult, String> neighCol = new TableColumn<>("Barrio");
        neighCol.setCellValueFactory(new PropertyValueFactory<>("neighbourhood"));
        neighCol.setPrefWidth(USE_COMPUTED_SIZE);
        neighCol.setMinWidth(COL_NEIGHBOURHOOD_MIN_WIDTH);
        neighCol.setMaxWidth(COL_NEIGHBOURHOOD_MAX_WIDTH);
        neighCol.setResizable(true);

        TableColumn<PropertyResult, String> typeCol = new TableColumn<>("Tipo");
        typeCol.setCellValueFactory(new PropertyValueFactory<>("propertyType"));
        typeCol.setPrefWidth(USE_COMPUTED_SIZE);
        typeCol.setMinWidth(COL_TYPE_MIN_WIDTH);
        typeCol.setMaxWidth(COL_TYPE_MAX_WIDTH);
        typeCol.setResizable(true);

        TableColumn<PropertyResult, Double> priceCol = new TableColumn<>("Precio");
        priceCol.setCellValueFactory(new PropertyValueFactory<>("price"));
        priceCol.setPrefWidth(USE_COMPUTED_SIZE);
        priceCol.setMinWidth(COL_PRICE_MIN_WIDTH); // Small minimum for numbers
        priceCol.setMaxWidth(COL_PRICE_MAX_WIDTH);
        priceCol.setResizable(true);
        priceCol.setCellFactory(column -> new TableCell<PropertyResult, Double>() {
            @Override
            protected void updateItem(Double price, boolean empty) {
                super.updateItem(price, empty);
                if (empty || price == null) {
                    setText(null);
                } else {
                    setText(String.format(FORMAT_PRICE, price));
                }
            }
        });

        TableColumn<PropertyResult, Double> ratingCol = new TableColumn<>("Rating");
        ratingCol.setCellValueFactory(new PropertyValueFactory<>("rating"));
        ratingCol.setPrefWidth(USE_COMPUTED_SIZE);
        ratingCol.setMinWidth(COL_RATING_MIN_WIDTH); // Small minimum for numbers
        ratingCol.setMaxWidth(COL_RATING_MAX_WIDTH);
        ratingCol.setResizable(true);
        ratingCol.setCellFactory(column -> new TableCell<PropertyResult, Double>() {
            @Override
            protected void updateItem(Double rating, boolean empty) {
                super.updateItem(rating, empty);
                if (empty || rating == null || rating == 0.0) {
                    setText(null);
                } else {
                    setText(String.format(FORMAT_RATING, rating));
                }
            }
        });

        TableColumn<PropertyResult, Integer> reviewsCol = new TableColumn<>("Reseñas");
        reviewsCol.setCellValueFactory(new PropertyValueFactory<>("reviews"));
        reviewsCol.setPrefWidth(USE_COMPUTED_SIZE);
        reviewsCol.setMinWidth(COL_REVIEWS_MIN_WIDTH); // Small minimum for numbers
        reviewsCol.setMaxWidth(COL_REVIEWS_MAX_WIDTH);
        reviewsCol.setResizable(true);
        reviewsCol.setCellFactory(column -> new TableCell<PropertyResult, Integer>() {
            @Override
            protected void updateItem(Integer reviews, boolean empty) {
                super.updateItem(reviews, empty);
                if (empty || reviews == null || reviews == 0) {
                    setText(null);
                } else {
                    setText(String.valueOf(reviews));
                }
            }
        });

        TableColumn<PropertyResult, Integer> bedroomsCol = new TableColumn<>("Habitaciones");
        bedroomsCol.setCellValueFactory(new PropertyValueFactory<>("bedrooms"));
        bedroomsCol.setPrefWidth(USE_COMPUTED_SIZE);
        bedroomsCol.setMinWidth(COL_BEDROOMS_MIN_WIDTH); // Small minimum for numbers
        bedroomsCol.setMaxWidth(COL_BEDROOMS_MAX_WIDTH);
        bedroomsCol.setResizable(true);
        bedroomsCol.setCellFactory(column -> new TableCell<PropertyResult, Integer>() {
            @Override
            protected void updateItem(Integer bedrooms, boolean empty) {
                super.updateItem(bedrooms, empty);
                if (empty || bedrooms == null || bedrooms == 0) {
                    setText(null);
                } else {
                    setText(String.valueOf(bedrooms));
                }
            }
        });

        TableColumn<PropertyResult, Integer> bathroomsCol = new TableColumn<>("Baños");
        bathroomsCol.setCellValueFactory(new PropertyValueFactory<>("bathrooms"));
        bathroomsCol.setPrefWidth(USE_COMPUTED_SIZE);
        bathroomsCol.setMinWidth(COL_BATHROOMS_MIN_WIDTH); // Very small minimum for single digits
        bathroomsCol.setMaxWidth(COL_BATHROOMS_MAX_WIDTH);
        bathroomsCol.setResizable(true);
        bathroomsCol.setCellFactory(column -> new TableCell<PropertyResult, Integer>() {
            @Override
            protected void updateItem(Integer bathrooms, boolean empty) {
                super.updateItem(bathrooms, empty);
                if (empty || bathrooms == null || bathrooms == 0) {
                    setText(null);
                } else {
                    setText(String.valueOf(bathrooms));
                }
            }
        });

        TableColumn<PropertyResult, String> urlCol = new TableColumn<>("URL");
        urlCol.setCellValueFactory(new PropertyValueFactory<>("listingUrl"));
        urlCol.setPrefWidth(USE_COMPUTED_SIZE);
        urlCol.setMinWidth(COL_URL_MIN_WIDTH); // Small minimum for "Ver" link
        urlCol.setMaxWidth(COL_URL_MAX_WIDTH);
        urlCol.setResizable(true);
        urlCol.setCellFactory(column -> new TableCell<PropertyResult, String>() {
            private final Hyperlink hyperlink = new Hyperlink();
            
            {
                hyperlink.setOnAction(e -> {
                    String url = (String) hyperlink.getUserData();
                    if (url != null && !url.isEmpty() && hostServices != null) {
                        hostServices.showDocument(url);
                    }
                });
            }
            
            @Override
            protected void updateItem(String url, boolean empty) {
                super.updateItem(url, empty);
                if (empty || url == null || url.isEmpty()) {
                    setGraphic(null);
                } else {
                    hyperlink.setText("Ver");
                    hyperlink.setUserData(url);
                    setGraphic(hyperlink);
                }
            }
        });

        TableColumn<PropertyResult, String> amenitiesCol = new TableColumn<>("Amenidades");
        amenitiesCol.setCellValueFactory(new PropertyValueFactory<>("amenities"));
        amenitiesCol.setPrefWidth(USE_COMPUTED_SIZE);
        amenitiesCol.setMinWidth(COL_AMENITIES_MIN_WIDTH);
        amenitiesCol.setMaxWidth(COL_AMENITIES_MAX_WIDTH);
        amenitiesCol.setResizable(true);
        amenitiesCol.setCellFactory(column -> new TableCell<PropertyResult, String>() {
            @Override
            protected void updateItem(String amenities, boolean empty) {
                super.updateItem(amenities, empty);
                if (empty || amenities == null || amenities.isEmpty()) {
                    setText(null);
                } else {
                    setText(amenities);
                    setWrapText(true);
                }
            }
        });

        resultsTable.getColumns().addAll(nameCol, descriptionCol, neighCol, typeCol, priceCol, ratingCol, reviewsCol, bedroomsCol, bathroomsCol, urlCol, amenitiesCol);

        // Configurar el TableView para ajustar la altura de las filas al contenido
        resultsTable.setRowFactory(tv -> {
            TableRow<PropertyResult> row = new TableRow<PropertyResult>() {
                @Override
                protected void updateItem(PropertyResult item, boolean empty) {
                    super.updateItem(item, empty);
                    if (empty || item == null) {
                        setPrefHeight(USE_COMPUTED_SIZE);
                    } else {
                        setPrefHeight(USE_COMPUTED_SIZE);
                    }
                }
            };
            return row;
        });

        return resultsTable;
    }

    /**
     * Barra de estado inferior.
     */
    private HBox createStatusBar() {
        HBox bar = new HBox(STATUS_BAR_SPACING);
        bar.setAlignment(Pos.CENTER_LEFT);
        bar.setPadding(new Insets(STATUS_BAR_PADDING_TOP, 0, 0, 0));

        statusLabel = new Label("Listo.");
        Label queryLabel = new Label();
        queryLabel.setStyle("-fx-font-size: " + FONT_SIZE_QUERY_LABEL + "px; -fx-text-fill: #666;");

        // Guardamos la referencia en la propia etiqueta de estado mediante userData para no
        // añadir más campos a la clase.
        statusLabel.setUserData(queryLabel);

        bar.getChildren().addAll(statusLabel, new Separator(Orientation.VERTICAL), queryLabel);

        return bar;
    }

    /**
     * Ejecuta la búsqueda en un hilo de fondo sencillo y actualiza la UI al terminar.
     * Inspirado en BusquedasLucene, ahora soporta múltiples campos y operadores.
     */
    private void executeSearch() {
        final String queryText = simpleQueryField.getText() != null
                ? simpleQueryField.getText().trim()
                : "";
        final String neighbourhood = neighbourhoodField != null && neighbourhoodField.getText() != null
                ? neighbourhoodField.getText().trim().toLowerCase()
                : "";
        final String minPriceText = minPriceField != null && minPriceField.getText() != null 
                ? minPriceField.getText().trim() 
                : "";
        final String maxPriceText = maxPriceField != null && maxPriceField.getText() != null 
                ? maxPriceField.getText().trim() 
                : "";
        final String ratingText = ratingField != null && ratingField.getText() != null
                ? ratingField.getText().trim()
                : "";
        final String reviewsText = reviewsField != null && reviewsField.getText() != null
                ? reviewsField.getText().trim()
                : "";
        final String bedroomsText = bedroomsField != null && bedroomsField.getText() != null
                ? bedroomsField.getText().trim()
                : "";
        final String bathroomsText = bathroomsField != null && bathroomsField.getText() != null
                ? bathroomsField.getText().trim()
                : "";
        final String amenityText = amenityField != null && amenityField.getText() != null
                ? amenityField.getText().trim()
                : "";
        final String propertyTypeText = propertyTypeField != null && propertyTypeField.getText() != null
                ? propertyTypeField.getText().trim()
                : "";
        final String latText = latField != null && latField.getText() != null
                ? latField.getText().trim()
                : "";
        final String lonText = lonField != null && lonField.getText() != null
                ? lonField.getText().trim()
                : "";
        final String radiusText = radiusField != null && radiusField.getText() != null
                ? radiusField.getText().trim()
                : "";

        statusLabel.setText("Buscando...");

        // Sin facetas activas al pulsar "Buscar"
        runSearchInBackground(queryText, neighbourhood, minPriceText, maxPriceText, 
                ratingText, reviewsText, bedroomsText, bathroomsText,
                amenityText, propertyTypeText, latText, lonText, radiusText, null);
    }

    /**
     * Ejecuta la búsqueda (con o sin faceta) en un hilo de fondo.
     * Inspirado en BusquedasLucene, ahora soporta múltiples campos y operadores.
     * Si activeFacets no es nulo, se aplica DrillDownQuery como en
     * BusquedasLucene.ejecutarQueryMegaCampoConFacetas (7.2).
     */
    private void runSearchInBackground(String queryText,
                                       String neighbourhood,
                                       String minPriceText,
                                       String maxPriceText,
                                       String ratingText,
                                       String reviewsText,
                                       String bedroomsText,
                                       String bathroomsText,
                                       String amenityText,
                                       String propertyTypeText,
                                       String latText,
                                       String lonText,
                                       String radiusText,
                                       Map<String, List<String>> activeFacets) {
        new Thread(() -> {
            long start = System.currentTimeMillis();
            List<PropertyResult> results = new ArrayList<>();
            Map<String, List<LabelAndValue>> facetsData = new LinkedHashMap<>();
            String luceneQueryText = "";
            String errorMessage = null;
            long totalHits = 0;

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

                // 3) Filtros por precio (DoublePoint) - soporta rango o operadores
                if (minPriceText != null && !minPriceText.isEmpty() && maxPriceText != null && !maxPriceText.isEmpty()) {
                    // Rango de precio (formato MIN-MAX)
                    Double minPrice = parseDoubleOrNull(minPriceText);
                    Double maxPrice = parseDoubleOrNull(maxPriceText);
                    if (minPrice != null || maxPrice != null) {
                        double min = minPrice != null ? minPrice : 0.0;
                        double max = maxPrice != null ? maxPrice : Double.MAX_VALUE;
                        Query priceQuery = DoublePoint.newRangeQuery("price", min, max);
                        queryBuilder.add(priceQuery, BooleanClause.Occur.FILTER);
                    }
                } else if (minPriceText != null && !minPriceText.isEmpty()) {
                    // Operador en minPriceText (ej: >=100, >100, <200, <=200, =150)
                    Query priceQuery = buildNumericQueryWithOperator("price", minPriceText, true);
                    if (priceQuery != null) {
                        queryBuilder.add(priceQuery, BooleanClause.Occur.FILTER);
                    }
                } else if (maxPriceText != null && !maxPriceText.isEmpty()) {
                    // Operador en maxPriceText
                    Query priceQuery = buildNumericQueryWithOperator("price", maxPriceText, true);
                    if (priceQuery != null) {
                        queryBuilder.add(priceQuery, BooleanClause.Occur.FILTER);
                    }
                }

                // 4) Filtro por rating (review_scores_rating) con operadores
                if (ratingText != null && !ratingText.isEmpty()) {
                    Query ratingQuery = buildNumericQueryWithOperator("review_scores_rating", ratingText, true);
                    if (ratingQuery != null) {
                        queryBuilder.add(ratingQuery, BooleanClause.Occur.FILTER);
                    }
                }

                // 5) Filtro por número de reseñas (number_of_reviews) con operadores
                if (reviewsText != null && !reviewsText.isEmpty()) {
                    Query reviewsQuery = buildNumericQueryWithOperator("number_of_reviews", reviewsText, false);
                    if (reviewsQuery != null) {
                        queryBuilder.add(reviewsQuery, BooleanClause.Occur.FILTER);
                    }
                }

                // 6) Filtro por habitaciones (bedrooms) con operadores
                if (bedroomsText != null && !bedroomsText.isEmpty()) {
                    Query bedroomsQuery = buildNumericQueryWithOperator("bedrooms", bedroomsText, false);
                    if (bedroomsQuery != null) {
                        queryBuilder.add(bedroomsQuery, BooleanClause.Occur.FILTER);
                    }
                }

                // 7) Filtro por baños (bathrooms) con operadores
                if (bathroomsText != null && !bathroomsText.isEmpty()) {
                    Query bathroomsQuery = buildNumericQueryWithOperator("bathrooms", bathroomsText, false);
                    if (bathroomsQuery != null) {
                        queryBuilder.add(bathroomsQuery, BooleanClause.Occur.FILTER);
                    }
                }

                // 8) Filtro por amenidad (amenity) - búsqueda textual
                if (amenityText != null && !amenityText.isEmpty()) {
                    QueryParser amenityParser = new QueryParser("amenity", analyzer);
                    Query amenityQuery = amenityParser.parse(amenityText);
                    queryBuilder.add(amenityQuery, BooleanClause.Occur.MUST);
                }

                // 9) Filtro por tipo de propiedad (property_type) - búsqueda textual
                if (propertyTypeText != null && !propertyTypeText.isEmpty()) {
                    QueryParser propertyTypeParser = new QueryParser("property_type", analyzer);
                    Query propertyTypeQuery = propertyTypeParser.parse(propertyTypeText.toLowerCase());
                    queryBuilder.add(propertyTypeQuery, BooleanClause.Occur.FILTER);
                }

                // 10) Búsqueda geográfica (lat, lon, radio)
                if (latText != null && !latText.isEmpty() && lonText != null && !lonText.isEmpty() 
                        && radiusText != null && !radiusText.isEmpty()) {
                    try {
                        double lat = Double.parseDouble(latText);
                        double lon = Double.parseDouble(lonText);
                        double radiusMeters = Double.parseDouble(radiusText);
                        if (radiusMeters > 0) {
                            Query geoQuery = LatLonPoint.newDistanceQuery(
                                    "location", lat, lon, radiusMeters);
                            queryBuilder.add(geoQuery, BooleanClause.Occur.FILTER);
                        }
                    } catch (NumberFormatException e) {
                        // Ignorar si los valores no son válidos
                    }
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
                    List<FacetResult> allDimsInitial = facetsInitial.getAllDims(FACET_DIMS_INITIAL_LIMIT);
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
                            List<FacetResult> allDimsAll = facetsAll.getAllDims(FACET_DIMS_ALL_LIMIT); // Obtener muchas para asegurar que tenemos todas
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
                        DrillSideways.DrillSidewaysResult dsResult = drillSideways.search(ddq, MAX_RESULTS);
                        
                        topDocs = dsResult.hits;
                        // Obtener el total de resultados usando reflexión para acceder al campo value
                        try {
                            java.lang.reflect.Field valueField = topDocs.totalHits.getClass().getDeclaredField("value");
                            valueField.setAccessible(true);
                            totalHits = valueField.getLong(topDocs.totalHits);
                        } catch (Exception e) {
                            // Fallback: usar el número de resultados mostrados
                            totalHits = topDocs.scoreDocs.length;
                        }
                        
                        // Obtener facetas del resultado de DrillSideways (mantiene conteos de todas las facetas)
                        Facets facets = dsResult.facets;
                        List<FacetResult> allDims = facets.getAllDims(FACET_DIMS_INITIAL_LIMIT);
                        if (allDims != null) {
                            for (FacetResult fr : allDims) {
                                if (fr != null && fr.dim != null) {
                                    facetsData.put(fr.dim, Arrays.asList(fr.labelValues));
                                }
                            }
                        }
                    } else {
                        // Sin facetas activas: búsqueda normal y recolección de facetas estándar
                        topDocs = searcher.search(baseQuery, MAX_RESULTS);
                        // Obtener el total de resultados usando reflexión para acceder al campo value
                        try {
                            java.lang.reflect.Field valueField = topDocs.totalHits.getClass().getDeclaredField("value");
                            valueField.setAccessible(true);
                            totalHits = valueField.getLong(topDocs.totalHits);
                        } catch (Exception e) {
                            // Fallback: usar el número de resultados mostrados
                            totalHits = topDocs.scoreDocs.length;
                        }
                        facetsData = availableFacets;
                    }

                    // Crear query para highlighting (solo si hay texto de búsqueda)
                    Query highlightQuery = null;
                    if (queryText != null && !queryText.isEmpty()) {
                        try {
                            QueryParser descriptionParser = new QueryParser("description", analyzer);
                            highlightQuery = descriptionParser.parse(queryText);
                        } catch (Exception e) {
                            // Si falla el parseo, no aplicar highlighting
                            highlightQuery = null;
                        }
                    }

                    // Procesar resultados de documentos
                    for (ScoreDoc sd : topDocs.scoreDocs) {
                        Document doc = searcher.storedFields().document(sd.doc);

                        String name = doc.get("name");
                        String neigh = doc.get("neighbourhood_cleansed_original");
                        String type = doc.get("property_type_original");
                        String description = doc.get("description");

                        // Aplicar highlighting a la descripción si hay query de texto
                        String descriptionHighlighted = description;
                        if (highlightQuery != null && description != null && !description.isEmpty()) {
                            descriptionHighlighted = applyHighlighting(description, highlightQuery, analyzer, "description");
                        }

                        Double price = null;
                        String priceStr = doc.get("price");
                        if (priceStr != null) {
                            try {
                                price = Double.parseDouble(priceStr);
                            } catch (NumberFormatException ignored) {
                            }
                        }

                        Double rating = null;
                        String ratingStr = doc.get("review_scores_rating");
                        if (ratingStr != null) {
                            try {
                                rating = Double.parseDouble(ratingStr);
                            } catch (NumberFormatException ignored) {
                            }
                        }

                        Integer reviews = null;
                        String reviewsStr = doc.get("number_of_reviews");
                        if (reviewsStr != null) {
                            try {
                                reviews = Integer.parseInt(reviewsStr);
                            } catch (NumberFormatException ignored) {
                            }
                        }

                        Integer bedrooms = null;
                        String bedroomsStr = doc.get("bedrooms");
                        if (bedroomsStr != null) {
                            try {
                                bedrooms = Integer.parseInt(bedroomsStr);
                            } catch (NumberFormatException ignored) {
                            }
                        }

                        Integer bathrooms = null;
                        String bathroomsStr = doc.get("bathrooms");
                        if (bathroomsStr != null) {
                            try {
                                // bathrooms puede ser decimal, pero lo tratamos como entero para la tabla
                                double bathroomsDouble = Double.parseDouble(bathroomsStr);
                                bathrooms = (int) Math.round(bathroomsDouble);
                            } catch (NumberFormatException ignored) {
                            }
                        }

                        String listingUrl = doc.get("listing_url");

                        // Extraer todas las amenidades (campo multivaluado)
                        String[] amenityValues = doc.getValues("amenity");
                        String amenitiesStr = "";
                        if (amenityValues != null && amenityValues.length > 0) {
                            amenitiesStr = String.join(", ", amenityValues);
                        }

                        results.add(new PropertyResult(0, name, neigh, type, price, rating, reviews, bedrooms, bathrooms, listingUrl, amenitiesStr, description, descriptionHighlighted));
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
            final long totalHitsFinal = totalHits;

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
                    String resultsText = totalHitsFinal > resultsFinal.size() 
                        ? "Encontrados " + totalHitsFinal + " resultados (mostrando " + resultsFinal.size() + ") en " + elapsed + " ms."
                        : "Encontrados " + resultsFinal.size() + " resultados en " + elapsed + " ms.";
                    statusLabel.setText(resultsText);
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

    /**
     * Helper: Parsea un string numérico con operador y devuelve el valor double.
     * Inspirado en BusquedasLucene.parseDoubleValue.
     * Soporta formatos: ">30.3", "<10", "10", "=100", ">=50.5", "<=20"
     * 
     * @param valorStr String con formato de operador y valor
     * @return El valor numérico extraído como double
     * @throws NumberFormatException Si el formato es inválido
     */
    private static double parseDoubleValue(String valorStr) throws NumberFormatException {
        String input = valorStr.trim();

        if (input.startsWith(">=")) {
            return Double.parseDouble(input.substring(2).trim());
        } else if (input.startsWith("<=")) {
            return Double.parseDouble(input.substring(2).trim());
        } else if (input.startsWith(">")) {
            return Double.parseDouble(input.substring(1).trim());
        } else if (input.startsWith("<")) {
            return Double.parseDouble(input.substring(1).trim());
        } else if (input.startsWith("=")) {
            return Double.parseDouble(input.substring(1).trim());
        } else {
            // Sin operador explícito, asumir que es solo el número
            return Double.parseDouble(input);
        }
    }

    /**
     * Construye una query numérica con operador (similar a BusquedasLucene.ejecutarQueryPrecioExacto).
     * Soporta operadores: >=, >, <, <=, =, o sin operador (igualdad).
     * 
     * @param fieldName Nombre del campo (ej: "price", "review_scores_rating", "bedrooms")
     * @param valorStr String con operador y valor (ej: ">=4.7", ">10", "=3")
     * @param isDouble true si es DoublePoint, false si es IntPoint
     * @return Query construida o null si el formato es inválido
     */
    private Query buildNumericQueryWithOperator(String fieldName, String valorStr, boolean isDouble) {
        if (valorStr == null || valorStr.trim().isEmpty()) {
            return null;
        }

        try {
            String input = valorStr.trim();
            Query query;

            if (isDouble) {
                double valor = parseDoubleValue(input);
                if (input.startsWith(">=")) {
                    query = DoublePoint.newRangeQuery(fieldName, valor, Double.MAX_VALUE);
                } else if (input.startsWith("<=")) {
                    query = DoublePoint.newRangeQuery(fieldName, DOUBLE_DEFAULT_MIN, valor);
                } else if (input.startsWith(">")) {
                    // Precio mayor: desde valor+epsilon hasta Double.MAX_VALUE
                    double min = valor + DOUBLE_EPSILON;
                    query = DoublePoint.newRangeQuery(fieldName, min, Double.MAX_VALUE);
                } else if (input.startsWith("<")) {
                    // Precio menor: desde 0 hasta valor-epsilon
                    double max = valor - DOUBLE_EPSILON;
                    query = DoublePoint.newRangeQuery(fieldName, DOUBLE_DEFAULT_MIN, max);
                } else if (input.startsWith("=")) {
                    query = DoublePoint.newRangeQuery(fieldName, valor, valor);
                } else {
                    // Sin operador explícito, asumir igualdad
                    query = DoublePoint.newRangeQuery(fieldName, valor, valor);
                }
            } else {
                // IntPoint
                double valorDouble = parseDoubleValue(input);
                int valor = (int) valorDouble;
                if (input.startsWith(">=")) {
                    query = IntPoint.newRangeQuery(fieldName, valor, Integer.MAX_VALUE);
                } else if (input.startsWith("<=")) {
                    query = IntPoint.newRangeQuery(fieldName, Integer.MIN_VALUE, valor);
                } else if (input.startsWith(">")) {
                    // Excluir el valor exacto
                    int min = valor + INT_EPSILON;
                    query = IntPoint.newRangeQuery(fieldName, min, Integer.MAX_VALUE);
                } else if (input.startsWith("<")) {
                    // Excluir el valor exacto
                    int max = valor - INT_EPSILON;
                    query = IntPoint.newRangeQuery(fieldName, Integer.MIN_VALUE, max);
                } else if (input.startsWith("=")) {
                    query = IntPoint.newRangeQuery(fieldName, valor, valor);
                } else {
                    // Sin operador explícito, asumir igualdad
                    query = IntPoint.newRangeQuery(fieldName, valor, valor);
                }
            }
            return query;
        } catch (NumberFormatException e) {
            return null;
        }
    }

    /**
     * Parsea texto con etiquetas <mark> y crea un TextFlow para mostrar en JavaFX.
     * 
     * @param highlightedText Texto con etiquetas <mark> para resaltar
     * @return TextFlow con texto normal y texto resaltado
     */
    private TextFlow parseHighlightedText(String highlightedText) {
        TextFlow textFlow = new TextFlow();
        
        if (highlightedText == null || highlightedText.isEmpty()) {
            return textFlow;
        }
        
        // Parsear manualmente las etiquetas <mark> y </mark>
        int pos = 0;
        boolean inHighlight = false;
        
        while (pos < highlightedText.length()) {
            if (highlightedText.startsWith(HTML_TAG_MARK_OPEN, pos)) {
                pos += HTML_TAG_MARK_OPEN_LENGTH; // Longitud de "<mark>"
                inHighlight = true;
            } else if (highlightedText.startsWith(HTML_TAG_MARK_CLOSE, pos)) {
                pos += HTML_TAG_MARK_CLOSE_LENGTH; // Longitud de "</mark>"
                inHighlight = false;
            } else {
                // Encontrar el siguiente tag o el final del texto
                int nextTag = highlightedText.length();
                int nextOpen = highlightedText.indexOf(HTML_TAG_MARK_OPEN, pos);
                int nextClose = highlightedText.indexOf(HTML_TAG_MARK_CLOSE, pos);
                
                if (nextOpen != -1 && nextOpen < nextTag) {
                    nextTag = nextOpen;
                }
                if (nextClose != -1 && nextClose < nextTag) {
                    nextTag = nextClose;
                }
                
                String textPart = highlightedText.substring(pos, nextTag);
                if (!textPart.isEmpty()) {
                    // Normalizar espacios y eliminar newlines que puedan causar altura extra
                    textPart = textPart.replaceAll("\\s+", " ").replaceAll("\\n+", " ").trim();
                    if (!textPart.isEmpty()) {
                        Text textNode = new Text(textPart);
                        if (inHighlight) {
                            // Texto resaltado: usar negrita y color oscuro para destacar
                            // Evitar StackPane/Region que causan problemas de wrapping en TextFlow
                            textNode.setStyle("-fx-font-weight: bold; -fx-fill: #B8860B;");
                        }
                        textFlow.getChildren().add(textNode);
                    }
                }
                pos = nextTag;
            }
        }
        
        return textFlow;
    }

    /**
     * Aplica highlighting a un texto usando la query de Lucene.
     * Retorna el texto con términos resaltados usando etiquetas HTML <mark>.
     * 
     * @param text Texto a resaltar
     * @param query Query de Lucene con los términos a resaltar
     * @param analyzer Analizador para tokenizar el texto
     * @param fieldName Nombre del campo (para el QueryScorer)
     * @return Texto resaltado con etiquetas <mark>, o el texto original si hay error
     */
    private String applyHighlighting(String text, Query query, Analyzer analyzer, String fieldName) {
        if (text == null || text.isEmpty() || query == null) {
            return text;
        }

        try {
            // Crear formatter HTML para resaltar con <mark>
            SimpleHTMLFormatter formatter = new SimpleHTMLFormatter("<mark>", "</mark>");
            
            // Crear scorer basado en la query
            QueryScorer scorer = new QueryScorer(query, fieldName);
            
            // Crear highlighter
            Highlighter highlighter = new Highlighter(formatter, scorer);
            highlighter.setTextFragmenter(new SimpleSpanFragmenter(scorer, Integer.MAX_VALUE));
            
            // Aplicar highlighting
            TokenStream tokenStream = analyzer.tokenStream(fieldName, new StringReader(text));
            String highlighted = highlighter.getBestFragment(tokenStream, text);
            
            // Si no hay fragmentos resaltados, retornar el texto original
            if (highlighted == null) {
                return text;
            }
            
            // Limpiar el texto resaltado: eliminar <br> tags y normalizar espacios
            // Esto previene que las filas se hagan excesivamente altas
            highlighted = highlighted
                .replaceAll("(?i)<br\\s*/?>", " ")  // Eliminar <br>, <br/>, <br />
                .replaceAll("\\s+", " ")             // Normalizar múltiples espacios a uno solo
                .replaceAll("\\n+", " ")             // Reemplazar newlines con espacios
                .trim();                             // Eliminar espacios al inicio/final
            
            return highlighted;
        } catch (Exception e) {
            // En caso de error, retornar el texto original
            return text;
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
        if (neighbourhoodField != null) neighbourhoodField.clear();
        if (minPriceField != null) minPriceField.clear();
        if (maxPriceField != null) maxPriceField.clear();
        if (ratingField != null) ratingField.clear();
        if (reviewsField != null) reviewsField.clear();
        if (bedroomsField != null) bedroomsField.clear();
        if (bathroomsField != null) bathroomsField.clear();
        if (amenityField != null) amenityField.clear();
        if (propertyTypeField != null) propertyTypeField.clear();
        if (latField != null) latField.clear();
        if (lonField != null) lonField.clear();
        if (radiusField != null) radiusField.clear();
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
            VBox mainPropertyTypeBox = new VBox(PROPERTY_TYPE_BOX_SPACING);
            
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
                VBox subFacetsBox = new VBox(SUBFACETS_BOX_SPACING);
                subFacetsBox.setPadding(new Insets(SUBFACETS_BOX_PADDING_TOP, 0, 0, SUBFACETS_BOX_PADDING_LEFT)); // Indentación para subfacetas
                
                // Extraer subfacetas activas para esta categoría
                List<String> activeForCategory = activeFacets != null ? 
                    extractActiveForCategory(activeFacets.get("property_type"), category) : null;
                
                // Crear lista de subfacetas que incluye las del resultado y las activas que no aparecen
                List<LabelAndValue> allSubTypes = new ArrayList<>(subTypes);
                Set<String> existingSubPaths = new HashSet<>();
                for (LabelAndValue lv : subTypes) {
                    existingSubPaths.add(category + "/" + lv.label);
                }
                
                // Agregar subfacetas activas que no están en los resultados
                if (activeForCategory != null) {
                    for (String activePath : activeForCategory) {
                        if (!existingSubPaths.contains(activePath)) {
                            // Extraer el label de la subfaceta del path completo
                            String[] parts = activePath.split("/", 2);
                            if (parts.length == 2 && parts[0].equals(category)) {
                                allSubTypes.add(new LabelAndValue(parts[1], 0L));
                            }
                        }
                    }
                }
                
                for (LabelAndValue lv : allSubTypes) {
                    String label = lv.label;
                    long count = lv.value.longValue();
                    String fullPath = category + "/" + label;
                    
                    // Mostrar count solo si es > 0, o mostrar "(aplicado)" si está activo pero tiene count 0
                    String countText = count > 0 ? " (" + count + ")" : 
                        (activeForCategory != null && activeForCategory.contains(fullPath) ? " (aplicado)" : " (0)");
                    
                    CheckBox cb = new CheckBox(label + countText);
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
                VBox categoryContainer = new VBox(CATEGORY_CONTAINER_SPACING);
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
            List<LabelAndValue> values = new ArrayList<>(entry.getValue());
            VBox box = new VBox(FACET_BOX_SPACING);

            List<String> activeForDim = activeFacets != null ? activeFacets.get(dim) : null;
            
            // Asegurar que las facetas activas siempre aparezcan en la lista, incluso si no están en los resultados
            // Esto previene que los filtros seleccionados desaparezcan de la interfaz
            if (activeForDim != null) {
                Set<String> existingLabels = new HashSet<>();
                for (LabelAndValue lv : values) {
                    existingLabels.add(lv.label);
                }
                // Agregar facetas activas que no están en los resultados con count 0
                for (String activeLabel : activeForDim) {
                    if (!existingLabels.contains(activeLabel)) {
                        values.add(new LabelAndValue(activeLabel, 0L));
                    }
                }
            }

            for (LabelAndValue lv : values) {
                String label = lv.label;
                long count = lv.value.longValue();
                
                // Traducir la etiqueta si hay traducción disponible
                Map<String, String> translations = labelTranslations.get(dim);
                String displayLabel = (translations != null && translations.containsKey(label)) 
                    ? translations.get(label) 
                    : label;
                
                // Mostrar count solo si es > 0, o mostrar "(aplicado)" si está activo pero tiene count 0
                String countText = count > 0 ? " (" + count + ")" : 
                    (activeForDim != null && activeForDim.contains(label) ? " (aplicado)" : " (0)");
                
                CheckBox cb = new CheckBox(displayLabel + countText);
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
        final String neighbourhood = neighbourhoodField != null && neighbourhoodField.getText() != null
                ? neighbourhoodField.getText().trim().toLowerCase()
                : "";
        final String minPriceText = minPriceField != null && minPriceField.getText() != null 
                ? minPriceField.getText().trim() 
                : "";
        final String maxPriceText = maxPriceField != null && maxPriceField.getText() != null 
                ? maxPriceField.getText().trim() 
                : "";
        final String ratingText = ratingField != null && ratingField.getText() != null
                ? ratingField.getText().trim()
                : "";
        final String reviewsText = reviewsField != null && reviewsField.getText() != null
                ? reviewsField.getText().trim()
                : "";
        final String bedroomsText = bedroomsField != null && bedroomsField.getText() != null
                ? bedroomsField.getText().trim()
                : "";
        final String bathroomsText = bathroomsField != null && bathroomsField.getText() != null
                ? bathroomsField.getText().trim()
                : "";
        final String amenityText = amenityField != null && amenityField.getText() != null
                ? amenityField.getText().trim()
                : "";
        final String propertyTypeText = propertyTypeField != null && propertyTypeField.getText() != null
                ? propertyTypeField.getText().trim()
                : "";
        final String latText = latField != null && latField.getText() != null
                ? latField.getText().trim()
                : "";
        final String lonText = lonField != null && lonField.getText() != null
                ? lonField.getText().trim()
                : "";
        final String radiusText = radiusField != null && radiusField.getText() != null
                ? radiusField.getText().trim()
                : "";

        statusLabel.setText("Aplicando filtros por facetas...");
        runSearchInBackground(queryText, neighbourhood, minPriceText, maxPriceText, 
                ratingText, reviewsText, bedroomsText, bathroomsText,
                amenityText, propertyTypeText, latText, lonText, radiusText, selectedFacets);
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
        final String neighbourhood = neighbourhoodField != null && neighbourhoodField.getText() != null
                ? neighbourhoodField.getText().trim().toLowerCase()
                : "";
        final String minPriceText = minPriceField != null && minPriceField.getText() != null 
                ? minPriceField.getText().trim() 
                : "";
        final String maxPriceText = maxPriceField != null && maxPriceField.getText() != null 
                ? maxPriceField.getText().trim() 
                : "";
        final String ratingText = ratingField != null && ratingField.getText() != null
                ? ratingField.getText().trim()
                : "";
        final String reviewsText = reviewsField != null && reviewsField.getText() != null
                ? reviewsField.getText().trim()
                : "";
        final String bedroomsText = bedroomsField != null && bedroomsField.getText() != null
                ? bedroomsField.getText().trim()
                : "";
        final String bathroomsText = bathroomsField != null && bathroomsField.getText() != null
                ? bathroomsField.getText().trim()
                : "";
        final String amenityText = amenityField != null && amenityField.getText() != null
                ? amenityField.getText().trim()
                : "";
        final String propertyTypeText = propertyTypeField != null && propertyTypeField.getText() != null
                ? propertyTypeField.getText().trim()
                : "";
        final String latText = latField != null && latField.getText() != null
                ? latField.getText().trim()
                : "";
        final String lonText = lonField != null && lonField.getText() != null
                ? lonField.getText().trim()
                : "";
        final String radiusText = radiusField != null && radiusField.getText() != null
                ? radiusField.getText().trim()
                : "";

        statusLabel.setText("Filtros de facetas limpiados.");
        runSearchInBackground(queryText, neighbourhood, minPriceText, maxPriceText, 
                ratingText, reviewsText, bedroomsText, bathroomsText,
                amenityText, propertyTypeText, latText, lonText, radiusText, null);
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