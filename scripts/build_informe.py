"""Genera el Informe Final PDF (~10 páginas) a partir de results.json y figuras."""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reports" / "results.json"
FIGURES = ROOT / "reports" / "figures"
OUT = ROOT / "reports" / "Informe_Final_Grupo33.pdf"


def pct(x, digits=1):
    return f"{100 * float(x):.{digits}f} %"


def num(x, digits=3):
    return f"{float(x):.{digits}f}"


def load_results():
    return json.loads(RESULTS.read_text(encoding="utf-8"))


def styles():
    base = getSampleStyleSheet()
    styles = {
        "cover_title": ParagraphStyle(
            "cover_title", parent=base["Title"], fontName="Helvetica-Bold",
            fontSize=20, leading=26, alignment=TA_CENTER, spaceAfter=12,
        ),
        "cover_sub": ParagraphStyle(
            "cover_sub", parent=base["Normal"], fontName="Helvetica",
            fontSize=12, leading=18, alignment=TA_CENTER, spaceAfter=6,
        ),
        "h1": ParagraphStyle(
            "h1", parent=base["Heading1"], fontName="Helvetica-Bold",
            fontSize=14, leading=20, spaceBefore=10, spaceAfter=8, textColor=colors.HexColor("#1F4E79"),
        ),
        "h2": ParagraphStyle(
            "h2", parent=base["Heading2"], fontName="Helvetica-Bold",
            fontSize=12, leading=16, spaceBefore=8, spaceAfter=6, textColor=colors.HexColor("#2E75B6"),
        ),
        "body": ParagraphStyle(
            "body", parent=base["Normal"], fontName="Helvetica",
            fontSize=11, leading=16.5, alignment=TA_JUSTIFY, spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "bullet", parent=base["Normal"], fontName="Helvetica",
            fontSize=11, leading=16.5, leftIndent=10, spaceAfter=3,
        ),
        "caption": ParagraphStyle(
            "caption", parent=base["Normal"], fontName="Helvetica-Oblique",
            fontSize=9, leading=12, alignment=TA_CENTER, spaceAfter=10, textColor=colors.grey,
        ),
        "footer": ParagraphStyle(
            "footer", parent=base["Normal"], fontName="Helvetica",
            fontSize=8, alignment=TA_CENTER, textColor=colors.grey,
        ),
    }
    return styles


def fig(name, width=15 * cm):
    path = FIGURES / name
    if not path.exists():
        return Paragraph(f"[Figura no disponible: {name}]", styles()["caption"])
    # Mantener proporción aproximada
    img = Image(str(path))
    img.drawWidth = width
    img.drawHeight = width * img.imageHeight / img.imageWidth
    if img.drawHeight > 8.5 * cm:
        scale = (8.5 * cm) / img.drawHeight
        img.drawWidth *= scale
        img.drawHeight *= scale
    return img


def table(data, col_widths=None):
    t = Table(data, colWidths=col_widths, hAlign="CENTER")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E79")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#AAAAAA")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EEF5FB")]),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return t


def best_model_block(split_payload):
    name = split_payload["best_model"]
    m = split_payload["models"][name]
    return name, m


def build():
    S = styles()
    data = load_results()
    eda = data["eda"]
    rand = data["random_split"]
    temp = data["temporal_split"]
    best_r_name, best_r = best_model_block(rand)
    best_t_name, best_t = best_model_block(temp)
    m50 = best_r["metrics_thr_0.5"]
    msel = best_r["metrics_selected"]
    cap = {row["k"]: row for row in best_r["captura_at_k"]}
    m50_t = best_t["metrics_thr_0.5"]

    story = []

    # Portada
    story.append(Spacer(1, 2.5 * cm))
    story.append(Paragraph("Diplomado en Ciencia de Datos y Análisis Avanzado", S["cover_sub"]))
    story.append(Paragraph("Unidad N° 9 — Proyecto Final Integrador", S["cover_sub"]))
    story.append(Spacer(1, 1.2 * cm))
    story.append(Paragraph("INFORME FINAL", S["cover_sub"]))
    story.append(Paragraph("Detección de Fraude<br/>en Tarjetas de Crédito", S["cover_title"]))
    story.append(Paragraph(
        "Modelo de clasificación supervisada para identificación<br/>de transacciones fraudulentas",
        S["cover_sub"],
    ))
    story.append(Spacer(1, 1.5 * cm))
    story.append(Paragraph("<b>Grupo 33</b>", S["cover_sub"]))
    story.append(Paragraph(
        "Mariana Chaile · Eugenia Salcedo Aisa · Sol Kidonakis<br/>Marco Garcia · Esteban Derghazarian",
        S["cover_sub"],
    ))
    story.append(Spacer(1, 1 * cm))
    story.append(Paragraph("Octubre 2026", S["cover_sub"]))
    story.append(PageBreak())

    # 1. Resumen ejecutivo
    story.append(Paragraph("1. Resumen Ejecutivo", S["h1"]))
    story.append(Paragraph(
        f"El fraude en tarjetas de crédito genera pérdidas globales superiores a USD 33.000 millones "
        f"anuales. Este proyecto construye un modelo de clasificación supervisada sobre el dataset "
        f"ULB (2013): <b>284.807</b> transacciones originales con <b>492</b> fraudes (0,172 %). "
        f"Tras eliminar {data['duplicates_removed']} duplicados exactos quedan "
        f"<b>{eda['n_rows']:,}</b> filas y <b>{eda['n_fraud']}</b> fraudes "
        f"({pct(eda['fraud_rate'], 3)}).",
        S["body"],
    ))
    story.append(Paragraph(
        f"Tras comparar Regresión Logística, Random Forest, XGBoost, LightGBM y XGBoost+SMOTE, "
        f"el modelo seleccionado en validación aleatoria es <b>{best_r_name}</b>, con "
        f"<b>PR-AUC = {num(m50['pr_auc'])}</b>, Recall = {pct(m50['recall'])} y "
        f"Precision = {pct(m50['precision'])} (umbral 0,5). "
        f"Si el equipo revisa las <b>100</b> operaciones de mayor riesgo, encuentra "
        f"<b>{cap[100]['frauds_found']}</b> fraudes; con las <b>500</b> de mayor riesgo, "
        f"<b>{cap[500]['frauds_found']}</b> fraudes (lift {num(cap[500]['lift'], 1)}× sobre la tasa base).",
        S["body"],
    ))
    story.append(Paragraph(
        f"La validación temporal (entrenar con operaciones anteriores y probar con posteriores) "
        f"confirma la robustez del enfoque: mejor modelo <b>{best_t_name}</b> con "
        f"PR-AUC = {num(m50_t['pr_auc'])}. El proyecto incorpora la devolución del tutor: "
        f"métricas principales PR-AUC / Recall / Precision, carga real de alertas y captura Top-K.",
        S["body"],
    ))

    # 2. Problema
    story.append(Paragraph("2. Definición del Problema y Relevancia", S["h1"]))
    story.append(Paragraph(
        "El área de Prevención de Fraude necesita priorizar revisiones humanas sobre un volumen "
        "masivo de transacciones. Un sistema de reglas estáticas suele generar muchas alertas "
        "imprecisas o dejar pasar patrones nuevos. El objetivo medible es clasificar cada "
        "transacción como fraudulenta (1) o legítima (0), maximizando la detección de fraudes "
        "y controlando la cantidad de alertas que el equipo puede revisar en la práctica.",
        S["body"],
    ))
    story.append(Paragraph(
        "<b>Hipótesis:</b> los patrones en monto, frecuencia temporal y componentes PCA permiten "
        "que un modelo de machine learning supere a reglas estáticas y priorice una cola de "
        "revisión accionable.",
        S["body"],
    ))
    story.append(Paragraph(
        "<b>Restricciones:</b> desbalance extremo; variables V1–V28 anonimizadas por PCA; "
        "dataset de solo 2 días; contexto académico sin streaming en tiempo real.",
        S["body"],
    ))

    # 3. Datos y metodología
    story.append(Paragraph("3. Datos y Metodología", S["h1"]))
    story.append(Paragraph("3.1 Fuente y calidad", S["h2"]))
    story.append(Paragraph(
        f"Dataset <i>Credit Card Fraud Detection</i> (ULB / Kaggle), licencia ODbL. "
        f"Tamaño: {eda['n_rows']:,} filas × {eda['n_cols']} columnas. "
        f"Nulos: {eda['n_nulls']}. Duplicados eliminados: {data['duplicates_removed']}. "
        f"Monto medio €{eda['amount_mean']:.1f} (mediana €{eda['amount_median']:.1f}). "
        f"Time entre {eda['time_min']:.0f} y {eda['time_max']:.0f} segundos.",
        S["body"],
    ))
    story.append(fig("01_class_distribution.png", 12 * cm))
    story.append(Paragraph("Figura 1. Distribución de clases (desbalance extremo).", S["caption"]))

    story.append(Paragraph("3.2 Preparación e ingeniería de variables", S["h2"]))
    story.append(Paragraph(
        "Se eliminaron filas exactamente duplicadas. Se creó <b>Amount_log = log1p(Amount)</b> "
        "para reducir sesgo. <b>Time</b> y <b>Amount_log</b> se escalan con StandardScaler "
        "dentro del Pipeline (sin leakage). V1–V28 se mantienen. El target es Class. "
        "Esta disciplina de pipeline es crítica: aplicar SMOTE o escalado sobre todo el dataset "
        "antes del split produciría data leakage y métricas infladas.",
        S["body"],
    ))
    story.append(fig("02_amount_by_class.png", 13 * cm))
    story.append(Paragraph("Figura 2. Distribución de Amount por clase (legítimas en escala log).", S["caption"]))
    story.append(fig("04_corr_with_class.png", 12 * cm))
    story.append(Paragraph("Figura 3. Variables con mayor |correlación| absoluta con Class.", S["caption"]))
    story.append(Paragraph(
        "El EDA confirma que un subconjunto de componentes PCA (V17, V14, V12, V10) concentra "
        "señal discriminativa, mientras que Amount por sí solo no separa bien las clases: "
        "muchos fraudes ocurren en montos bajos o medios. Time muestra concentraciones distintas "
        "entre legítimas y fraudes, lo que motiva la validación temporal.",
        S["body"],
    ))

    story.append(Paragraph("3.3 Metodología CRISP-DM y validación", S["h2"]))
    story.append(Paragraph(
        "Se siguen las fases CRISP-DM. Se comparan dos esquemas de validación: "
        "(A) holdout estratificado 80/20 aleatorio; (B) split temporal por Time "
        "(80 % operaciones anteriores → train; 20 % posteriores → test). "
        "Métricas principales: PR-AUC, Recall y Precision. Complementarias: F1, ROC-AUC, "
        "matriz de confusión, captura Top-K y volumen de alertas por umbral.",
        S["body"],
    ))

    # 4. Modelado
    story.append(Paragraph("4. Modelado y Evaluación", S["h1"]))
    story.append(Paragraph(
        "Se evaluaron cinco configuraciones: Regresión Logística (baseline, class_weight=balanced), "
        "Random Forest, XGBoost (scale_pos_weight), LightGBM (is_unbalance) y XGBoost+SMOTE "
        "dentro del pipeline.",
        S["body"],
    ))

    # Tabla comparativa aleatorio
    story.append(Paragraph("4.1 Resultados — split aleatorio", S["h2"]))
    header = ["Modelo", "PR-AUC", "Recall", "Precision", "F1", "Alertas", "Fraud@100", "Fraud@500"]
    rows = [header]
    for name, m in rand["models"].items():
        m0 = m["metrics_thr_0.5"]
        c = {r["k"]: r for r in m["captura_at_k"]}
        rows.append([
            name,
            num(m0["pr_auc"]),
            pct(m0["recall"], 1),
            pct(m0["precision"], 1),
            num(m0["f1"]),
            str(m0["alerts"]),
            str(c[100]["frauds_found"]),
            str(c[500]["frauds_found"]),
        ])
    story.append(table(rows, col_widths=[3.2*cm, 1.8*cm, 1.8*cm, 2*cm, 1.5*cm, 1.6*cm, 1.8*cm, 1.8*cm]))
    story.append(Paragraph(
        f"Tabla 1. Comparativa en holdout aleatorio (umbral 0,5). Mejor modelo: <b>{best_r_name}</b>.",
        S["caption"],
    ))

    story.append(fig("06_pr_curve_aleatorio.png", 10 * cm))
    story.append(Paragraph("Figura 4. Curva Precision-Recall del modelo seleccionado.", S["caption"]))
    story.append(Paragraph(
        f"Observación clave de la comparación: LightGBM con <i>is_unbalance</i> obtiene un PR-AUC "
        f"muy bajo pese a un Recall razonable, lo que ilustra por qué <b>PR-AUC</b> es la métrica "
        f"de ranking preferida frente a Accuracy o un Recall aislado. XGBoost+SMOTE mejora el "
        f"Recall respecto de XGBoost puro, pero degrada fuerte la Precision (muchas más alertas), "
        f"por lo que no se elige como modelo final.",
        S["body"],
    ))

    story.append(Paragraph("4.2 Métrica de negocio: captura Top-K", S["h2"]))
    story.append(Paragraph(
        f"Respondiendo a la devolución del tutor: si el equipo revisa las 100 operaciones de "
        f"mayor score, captura <b>{cap[100]['frauds_found']}</b> fraudes "
        f"(Precision@100 = {pct(cap[100]['precision_at_k'])}; "
        f"Recall@100 = {pct(cap[100]['recall_at_k'])}). "
        f"Con 500 revisiones: <b>{cap[500]['frauds_found']}</b> fraudes "
        f"(lift {num(cap[500]['lift'], 1)}×).",
        S["body"],
    ))
    story.append(fig("08_captura_topk_aleatorio.png", 13 * cm))
    story.append(Paragraph("Figura 5. Fraudes encontrados vs operaciones revisadas (Top-K).", S["caption"]))

    story.append(Paragraph("4.3 Precision, alertas y umbral operativo", S["h2"]))
    story.append(Paragraph(
        f"Con tan pocos fraudes, un FPR del 5 % sobre ~{m50['n_neg']:,} legítimas del holdout "
        f"implicaría miles de alertas falsas. Por eso el umbral no se fija solo por FPR. "
        f"Con la regla de negocio (Precision ≥ 50 % y tope de alertas), el umbral seleccionado "
        f"es <b>{msel['threshold']:.2f}</b>: Recall {pct(msel['recall'])}, "
        f"Precision {pct(msel['precision'])}, alertas a revisar = <b>{msel['alerts']}</b> "
        f"(TP={msel['tp']}, FP={msel['fp']}).",
        S["body"],
    ))
    story.append(fig("09_workload_aleatorio.png", 13 * cm))
    story.append(Paragraph("Figura 6. Precision y Recall en función de la carga de alertas.", S["caption"]))
    story.append(fig("07_confusion_aleatorio.png", 9 * cm))
    story.append(Paragraph("Figura 7. Matriz de confusión con umbral seleccionado.", S["caption"]))

    story.append(Paragraph("4.4 Validación temporal vs aleatoria", S["h2"]))
    story.append(Paragraph(
        f"En el split temporal (train hasta t={temp['meta'].get('train_time_max', '—')}, "
        f"test desde t={temp['meta'].get('test_time_min', '—')}; "
        f"{temp['meta'].get('test_fraud', '—')} fraudes en test), el mejor modelo es "
        f"<b>{best_t_name}</b> con PR-AUC={num(m50_t['pr_auc'])}, "
        f"Recall={pct(m50_t['recall'])} y Precision={pct(m50_t['precision'])}. "
        f"La caída o variación respecto del split aleatorio ilustra el riesgo de sobreestimar "
        f"performance cuando el futuro se mezcla aleatoriamente con el pasado.",
        S["body"],
    ))
    story.append(fig("11_comparison_pr_auc.png", 14 * cm))
    story.append(Paragraph("Figura 8. PR-AUC por modelo: aleatorio vs temporal.", S["caption"]))
    # Tabla temporal resumida
    header_t = ["Modelo", "PR-AUC", "Recall", "Precision", "Fraud@100", "Fraud@500"]
    rows_t = [header_t]
    for name, m in temp["models"].items():
        m0 = m["metrics_thr_0.5"]
        c = {r["k"]: r for r in m["captura_at_k"]}
        rows_t.append([
            name, num(m0["pr_auc"]), pct(m0["recall"], 1), pct(m0["precision"], 1),
            str(c[100]["frauds_found"]), str(c[500]["frauds_found"]),
        ])
    story.append(table(rows_t, col_widths=[3.4*cm, 2*cm, 2*cm, 2.2*cm, 2.2*cm, 2.2*cm]))
    story.append(Paragraph(
        f"Tabla 2. Comparativa en holdout temporal. Mejor modelo: <b>{best_t_name}</b>.",
        S["caption"],
    ))

    story.append(Paragraph("4.5 Interpretabilidad", S["h2"]))
    story.append(fig("10_importance_aleatorio.png", 12 * cm))
    story.append(Paragraph(
        "Figura 9. Importancia de variables del modelo final (limitada por anonimización PCA).",
        S["caption"],
    ))
    story.append(Paragraph(
        "Aunque las importancias ayudan a priorizar componentes PCA, la anonimización limita "
        "la explicación causal ante auditorías. En un entorno productivo convendría complementar "
        "con señales interpretables (país, comercio, dispositivo, historial del cliente).",
        S["body"],
    ))

    # 5. Impacto
    story.append(Paragraph("5. Impacto en el Negocio", S["h1"]))
    story.append(Paragraph(
        "El valor del modelo no es solo “detectar más fraude”, sino <b>ordenar la cola de revisión</b>: "
        "el equipo puede concentrar esfuerzo en las 100–500 operaciones más riesgosas y recuperar "
        "una fracción alta de fraudes con un costo operativo acotado. Esto reduce pérdidas por "
        "fraudes no detectados y evita el desgaste de miles de falsas alarmas que produciría "
        "un umbral basado únicamente en FPR.",
        S["body"],
    ))
    story.append(Paragraph("5.1 Traducción a operación diaria", S["h2"]))
    story.append(Paragraph(
        f"En el holdout aleatorio hay {m50['n_pos']} fraudes reales. Con XGBoost, revisar las "
        f"100 operaciones de mayor score recupera {cap[100]['frauds_found']} fraudes "
        f"({pct(cap[100]['recall_at_k'])} del total del holdout). Ampliar a 500 revisiones "
        f"sube la captura a {cap[500]['frauds_found']} fraudes ({pct(cap[500]['recall_at_k'])}), "
        f"con Precision@500 = {pct(cap[500]['precision_at_k'])}. Ese trade-off es transparente "
        f"para un gerente de fraude: el costo es “horas de analista”, el beneficio es “fraudes "
        f"evitados / recuperados”.",
        S["body"],
    ))
    story.append(Paragraph(
        f"Con el umbral operativo {msel['threshold']:.2f}, el sistema emite solo "
        f"<b>{msel['alerts']} alertas</b> (TP={msel['tp']}, FP={msel['fp']}). "
        f"Comparado con un FPR del 5 % — que generaría aproximadamente "
        f"{int(0.05 * m50['n_neg']):,} alertas falsas solo en el holdout — la reducción de "
        f"ruido operativo es de órdenes de magnitud.",
        S["body"],
    ))
    story.append(Paragraph("5.2 Riesgos, limitaciones y escalabilidad", S["h2"]))
    story.append(Paragraph(
        "<b>Riesgos y limitaciones:</b> representatividad temporal (solo 2 días de 2013); "
        "poca interpretabilidad de V1–V28 ante auditoría; posible drift de patrones de fraude; "
        "umbral optimizado sobre un holdout — requiere monitoreo post-despliegue. "
        "Además, el dataset no incluye contexto de cliente ni comercio, por lo que el modelo "
        "aprende correlaciones de componentes PCA que pueden no transferir a otros emisores.",
        S["body"],
    ))
    story.append(Paragraph(
        "<b>Escalabilidad y próximos pasos:</b> (1) scoring batch o near-real-time sobre el "
        "motor de autorizaciones; (2) feedback loop del analista para reentrenar con etiquetas "
        "nuevas; (3) calibración de probabilidades; (4) features adicionales (dispositivo, geo, "
        "historial del cliente, merchant risk); (5) monitoreo de drift de score y de lift@K.",
        S["body"],
    ))
    story.append(Paragraph(
        "<b>Benchmark de reglas estáticas (cualitativo):</b> un sistema de reglas típicamente "
        "prioriza montos altos, países de riesgo o velocidad de compra. En este dataset muchos "
        "fraudes no se distinguen solo por Amount; el modelo captura patrones multivariados en "
        "PCA y prioriza mejor la cola. La recomendación no es reemplazar por completo las reglas, "
        "sino combinarlas con el score ML como capa de priorización.",
        S["body"],
    ))

    # 6. Ética
    story.append(Paragraph("6. Consideraciones éticas y de privacidad", S["h1"]))
    story.append(Paragraph(
        "Los datos están anonimizados mediante PCA; no se manipulan PII. Un falso positivo puede "
        "bloquear a un cliente legítimo (fricción y reputación); un falso negativo implica pérdida "
        "financiera. El umbral se elige de forma explícita como decisión de negocio, no como "
        "default técnico. Uso académico bajo licencia ODbL.",
        S["body"],
    ))
    story.append(Paragraph(
        "Adicionalmente, se recomienda: (a) registrar quién revisó cada alerta y el resultado "
        "(fraude confirmado / falso positivo) para auditoría; (b) evitar discriminación indirecta "
        "si en el futuro se agregan variables demográficas o geográficas sensibles; "
        "(c) comunicar al cliente, cuando corresponda, mecanismos de apelación ante bloqueos; "
        "(d) limitar el acceso a scores y features al personal autorizado (need-to-know).",
        S["body"],
    ))
    story.append(Paragraph(
        "El equipo asume que este trabajo es un prototipo académico y no un sistema de decisión "
        "automatizada en producción: cualquier despliegue real requeriría validación legal "
        "(p. ej. lineamientos de modelos de riesgo), pruebas con datos propios del emisor y "
        "supervisión humana en la cola de alertas.",
        S["body"],
    ))

    # 7. Conclusiones
    story.append(Paragraph("7. Conclusiones y Recomendaciones", S["h1"]))
    bullets = [
        f"Seleccionar <b>{best_r_name}</b> como modelo productivo candidato (mejor PR-AUC en holdout aleatorio).",
        "Reportar siempre PR-AUC, Recall, Precision y volumen de alertas; no guiarse solo por FPR o Accuracy.",
        "Usar captura Top-100 / Top-500 como KPI operativo de la cola de revisión.",
        "Validar con esquema temporal antes de desplegar; el split aleatorio puede ser optimista.",
        "Implementar monitoreo continuo del umbral y del lift@K con feedback del equipo de fraude.",
    ]
    for b in bullets:
        story.append(Paragraph(f"• {b}", S["bullet"]))

    story.append(PageBreak())
    story.append(Paragraph("Anexos", S["h1"]))
    story.append(Paragraph(
        "Código reproducible en el repositorio GitHub del Grupo 33 "
        "(notebook + módulo src + requirements). Figuras adicionales en reports/figures/. "
        "Resultados numéricos serializados en reports/results.json.",
        S["body"],
    ))
    story.append(fig("03_time_by_class.png", 13 * cm))
    story.append(Paragraph("Anexo A. Distribución temporal por clase.", S["caption"]))
    story.append(fig("05_boxplots_by_class.png", 14 * cm))
    story.append(Paragraph("Anexo B. Boxplots de variables discriminativas por clase.", S["caption"]))
    story.append(fig("08_captura_topk_temporal.png", 13 * cm))
    story.append(Paragraph("Anexo C. Captura Top-K en validación temporal.", S["caption"]))

    def _footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawCentredString(A4[0] / 2, 1.2 * cm, f"Grupo 33 — Detección de Fraude | Página {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title="Informe Final — Detección de Fraude — Grupo 33",
        author="Grupo 33",
    )
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    print(f"Informe generado: {OUT}")


if __name__ == "__main__":
    build()
