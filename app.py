
import streamlit as st
import pandas as pd
import plotly.express as px

from src.data_utils import load_data, dataset_overview, normalize_columns
from src.enrichment import add_web_mining_enrichment, enrichment_dictionary
from src.neural_models import run_neural_benchmark
from src.association_rules import build_transactions, mine_rules
from src.eda import class_distribution, numeric_summary

st.set_page_config(
    page_title="Advanced Obesity Data Mining Dashboard",
    layout="wide"
)

DATA_PATH = "data/data.csv"

@st.cache_data
def get_data():
    return load_data(DATA_PATH)

def main():
    st.title("Advanced Obesity Data Mining Dashboard")
    st.caption(
        "Dashboard para el artículo científico: Web Mining Enrichment, "
        "Neural Networks Benchmarking y Association Rules sobre datos de obesidad."
    )

    df = get_data()
    target_col = "NObeyesdad"

    st.sidebar.header("Configuración general")
    module = st.sidebar.radio(
        "Módulo",
        [
            "1. Exploratory / Baseline",
            "2. Web Mining Enrichment",
            "3. Neural Network Benchmarking",
            "4. Association Rules"
        ],
        key="main_module"
    )

    if module == "1. Exploratory / Baseline":
        st.header("1. Exploratory / Baseline")

        c1, c2, c3 = st.columns(3)
        overview = dataset_overview(df)
        c1.metric("Filas", overview["rows"])
        c2.metric("Columnas", overview["columns"])
        c3.metric("Duplicados", overview["duplicates"])

        with st.expander("Vista del dataset", expanded=True):
            st.dataframe(df.head(30), use_container_width=True)

        st.subheader("Distribución de clases")
        dist = class_distribution(df, target_col)
        st.dataframe(dist, use_container_width=True)
        fig = px.bar(dist, x=target_col, y="count", title="Class distribution: NObeyesdad")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Resumen numérico")
        st.dataframe(numeric_summary(df), use_container_width=True)

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        selected_num = st.multiselect(
            "Variables numéricas para boxplot",
            numeric_cols,
            default=numeric_cols[:4],
            key="eda_numeric_cols"
        )
        if selected_num:
            long_df = df[selected_num].melt(var_name="variable", value_name="value")
            fig = px.box(long_df, x="variable", y="value", title="Boxplots of selected numeric variables")
            st.plotly_chart(fig, use_container_width=True)

    elif module == "2. Web Mining Enrichment":
        st.header("2. Web Mining Enrichment como enriquecimiento de datos")
        st.write(
            "Este módulo operacionaliza el Web Mining como una capa de enriquecimiento "
            "basada en conocimiento externo: las variables conductuales del dataset se mapean "
            "a factores de riesgo nutricional, sedentarismo, antecedentes familiares y estilo de vida."
        )

        enriched = add_web_mining_enrichment(df)

        st.subheader("Diccionario de enriquecimiento")
        st.dataframe(enrichment_dictionary(), use_container_width=True)

        st.subheader("Variables enriquecidas generadas")
        new_cols = [
            "dietary_risk_score",
            "physical_inactivity_score",
            "screen_time_risk_score",
            "substance_risk_score",
            "mobility_risk_score",
            "hereditary_risk_score",
            "web_mining_enrichment_score",
            "web_mining_risk_profile"
        ]
        st.dataframe(enriched[new_cols + [target_col]].head(30), use_container_width=True)

        st.subheader("Distribución del perfil enriquecido")
        profile_dist = (
            enriched["web_mining_risk_profile"]
            .value_counts()
            .reset_index()
        )
        profile_dist.columns = ["profile", "count"]
        fig = px.bar(profile_dist, x="profile", y="count", title="Web Mining Enrichment Risk Profiles")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Relación entre perfil enriquecido y clase de obesidad")
        ctab = pd.crosstab(
            enriched["web_mining_risk_profile"],
            enriched[target_col],
            normalize="index"
        ).round(3)
        st.dataframe(ctab, use_container_width=True)

    elif module == "3. Neural Network Benchmarking":
        st.header("3. Neural Network Benchmarking")

        use_enrichment = st.sidebar.checkbox(
            "Usar variables enriquecidas por Web Mining",
            value=True,
            key="nn_use_enrichment"
        )
        test_size = st.sidebar.slider(
            "Test size",
            min_value=0.15,
            max_value=0.40,
            value=0.20,
            step=0.05,
            key="nn_test_size"
        )
        max_iter = st.sidebar.slider(
            "Máximo de iteraciones",
            min_value=100,
            max_value=800,
            value=300,
            step=100,
            key="nn_max_iter"
        )

        data_model = add_web_mining_enrichment(df) if use_enrichment else df.copy()

        st.write(
            "Se comparan cinco configuraciones defendibles de redes neuronales tipo MLP "
            "para clasificación multiclase de `NObeyesdad`."
        )

        run = st.button("Ejecutar benchmark neuronal", key="run_nn")
        if run:
            with st.spinner("Entrenando redes neuronales..."):
                results_df, details = run_neural_benchmark(
                    data_model,
                    target_col=target_col,
                    test_size=test_size,
                    max_iter=max_iter
                )

            st.success("Benchmark neuronal completado.")
            st.subheader("Tabla comparativa")
            st.dataframe(results_df, use_container_width=True)

            best = results_df.iloc[0]
            st.info(f"Mejor configuración por F1 weighted: **{best['model']}** ({best['f1_weighted']:.4f})")

            csv = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Descargar resultados CSV",
                data=csv,
                file_name="neural_network_benchmark.csv",
                mime="text/csv",
                key="download_nn_results"
            )

            selected_model = st.selectbox(
                "Ver matriz de confusión",
                options=results_df["model"].tolist(),
                key="nn_detail_model"
            )
            cm = details[selected_model]["confusion_matrix"]
            labels = details[selected_model]["labels"]
            cm_df = pd.DataFrame(cm, index=labels, columns=labels)
            fig = px.imshow(cm_df, text_auto=True, title=f"Confusion Matrix - {selected_model}")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Reporte de clasificación")
            st.dataframe(details[selected_model]["classification_report"], use_container_width=True)

    elif module == "4. Association Rules":
        st.header("4. Association Rules")

        st.write(
            "Este módulo transforma cada paciente en una transacción de ítems clínicos/conductuales "
            "y aplica reglas de asociación para descubrir patrones interpretables."
        )

        min_support = st.sidebar.slider("Min support", 0.01, 0.30, 0.05, 0.01, key="ar_support")
        min_confidence = st.sidebar.slider("Min confidence", 0.10, 0.95, 0.50, 0.05, key="ar_conf")
        min_lift = st.sidebar.slider("Min lift", 1.00, 5.00, 1.10, 0.10, key="ar_lift")

        enriched = add_web_mining_enrichment(df)
        transactions = build_transactions(enriched, target_col=target_col)

        with st.expander("Ejemplo de transacciones", expanded=False):
            st.write(transactions[:10])

        run_ar = st.button("Ejecutar reglas de asociación", key="run_ar")
        if run_ar:
            with st.spinner("Minando reglas de asociación..."):
                rules = mine_rules(
                    transactions,
                    min_support=min_support,
                    min_confidence=min_confidence,
                    min_lift=min_lift
                )

            if rules.empty:
                st.warning("No se encontraron reglas con esos umbrales. Reduce support/confidence/lift.")
            else:
                st.success(f"Se encontraron {len(rules)} reglas.")
                st.dataframe(rules, use_container_width=True)

                csv = rules.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Descargar reglas CSV",
                    data=csv,
                    file_name="association_rules_obesity.csv",
                    mime="text/csv",
                    key="download_rules"
                )

                top = rules.head(15).copy()
                top["rule"] = top["antecedents"].astype(str) + " -> " + top["consequents"].astype(str)
                fig = px.bar(top, x="lift", y="rule", orientation="h", title="Top rules by lift")
                st.plotly_chart(fig, use_container_width=True)

    else:
        st.header("5. Comparative Conclusions")
        st.write(
            "Este módulo consolida la lectura metodológica del dashboard: "
            "EDA para comprender la base, Web Mining Enrichment para crear variables interpretables, "
            "redes neuronales para benchmarking predictivo avanzado y reglas de asociación para extracción de conocimiento."
        )

        st.markdown(
            """
            **Lectura esperada para el artículo:**

            1. El enriquecimiento web/documental permite transformar variables conductuales en perfiles de riesgo.
            2. Las redes neuronales evalúan si configuraciones más complejas mejoran la predicción multiclase de obesidad.
            3. Las reglas de asociación producen patrones interpretables entre hábitos, antecedentes y niveles de obesidad.
            4. La comparación entre módulos fortalece el artículo porque combina predicción, enriquecimiento e interpretación.
            """
        )

        st.info(
            "Después de correr los módulos, exporta las tablas CSV y usa las figuras principales "
            "en Analysis and Results del artículo."
        )

if __name__ == "__main__":
    main()
