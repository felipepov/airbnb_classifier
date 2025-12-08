# Compilar con Maven

## Prerequisitos

1. Instalar Maven:
```bash
sudo apt install maven
# O descargar desde: https://maven.apache.org/download.cgi
```

2. Verificar instalación:
```bash
mvn --version
```

## Compilar y crear JAR

### 1. Compilar el proyecto:
```bash
mvn clean compile
```

### 2. Crear el JAR ejecutable (fat JAR con todas las dependencias):
```bash
mvn clean package
```

Esto generará el archivo `target/airbnb-indexer.jar` que contiene todas las dependencias incluidas.

## Ejecutar el JAR

Una vez creado el JAR, puedes ejecutarlo directamente sin necesidad de classpath:

```bash
java -jar target/airbnb-indexer.jar --input example_listings.csv --index-root ./index_root --mode rebuild --force
```

## Ventajas del JAR generado

- ✅ **Todo incluido**: No necesitas `lib/` ni `lucene-10.3.1/modules/` en el classpath
- ✅ **Portable**: Un solo archivo JAR que puedes copiar a cualquier máquina con Java 11+
- ✅ **Fácil de distribuir**: Solo necesitas el JAR y ejecutarlo con `java -jar`
- ✅ **Sin dependencias externas**: Todas las dependencias están dentro del JAR

## Estructura del proyecto

```
P3/
├── src/main/java/
│   └── AirbnbIndexador.java
├── pom.xml
├── target/
│   └── airbnb-indexer.jar  (generado por Maven)
└── ...
```

## Nota sobre lib/

Una vez que uses el JAR generado con Maven, ya **no necesitas** la carpeta `lib/` porque Gson está incluido en el JAR.

Pero puedes mantenerla si prefieres seguir usando el método manual de compilación.

---

## Compilar y ejecutar AirbnbSearchApp (Interfaz JavaFX)

`AirbnbSearchApp` es una aplicación JavaFX que proporciona una interfaz gráfica para realizar búsquedas sobre el índice de propiedades de Airbnb.

### Prerequisitos

- Java 21 o superior
- Maven instalado
- Índice de propiedades creado (usando `AirbnbIndexador`)

### Compilar

El proyecto ya está configurado en el `pom.xml`. Para compilar:

```bash
mvn clean compile
```

### Ejecutar con JavaFX Maven Plugin (Recomendado)

La forma más sencilla de ejecutar la aplicación es usando el plugin JavaFX de Maven:

```bash
mvn javafx:run
```

Para especificar una ruta personalizada del índice:

```bash
mvn javafx:run -Djavafx.args="--index-root ./index_root"
```

### Ejecutar directamente con java

Si prefieres ejecutar directamente con `java`, necesitas incluir los módulos de JavaFX:

```bash
# Desde el directorio raíz del proyecto
java --module-path /path/to/javafx-sdk/lib \
     --add-modules javafx.controls,javafx.graphics \
     -cp target/classes:target/dependency/* \
     AirbnbSearchApp --index-root ./index_root
```

**Nota:** Reemplaza `/path/to/javafx-sdk/lib` con la ruta real a tu instalación de JavaFX SDK, o usa las dependencias de Maven si están disponibles.

### Argumentos de línea de comandos

La aplicación acepta los siguientes argumentos:

- `--index-root <ruta>`: Especifica la ruta raíz donde se encuentran los índices (por defecto: `./index_root`)
  - Ejemplo: `--index-root ./index_root`
  - La aplicación buscará los índices en:
    - `{index-root}/index_properties/` (índice de propiedades)
    - `{index-root}/taxo_properties/` (taxonomía de propiedades para facetas)

### Ejemplo completo

```bash
# 1. Compilar el proyecto
mvn clean compile

# 2. Ejecutar la aplicación con el plugin JavaFX
mvn javafx:run -Djavafx.args="--index-root ./index_root"
```

### Características de la aplicación

- **Búsqueda simple**: Campo de texto para búsquedas en el mega campo "contents"
- **Búsqueda avanzada**: Filtros por precio, rating, reseñas, habitaciones, baños, amenidades, tipo de propiedad, barrio y geolocalización
- **Filtros por facetas**: Panel lateral con facetas interactivas para refinar búsquedas
- **Resultados destacados**: Resaltado de términos de búsqueda en las descripciones
- **Tabla de resultados**: Visualización de propiedades con múltiples columnas (nombre, descripción, barrio, tipo, precio, rating, etc.)

