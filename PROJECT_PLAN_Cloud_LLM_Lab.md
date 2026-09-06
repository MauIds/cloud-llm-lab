# Cloud LLM Lab — Plan Maestro del Proyecto

## Build, Fine-tune and Deploy Language Models from First Principles

**Proyecto principal:** MentorLM  
**Proyecto derivado:** TinyGPT  
**Infraestructura de cómputo:** Modal  
**Modelo base propuesto:** Qwen3.5-4B  
**Método de adaptación:** Supervised Fine-Tuning (SFT) + BF16 LoRA  
**Objetivo:** aprender de verdad cómo se construye, modifica, evalúa y despliega un LLM moderno, terminando con un proyecto sólido, documentado y compartible.

---

# 1. Visión del proyecto

Este proyecto no busca simplemente crear otro chatbot.

El objetivo es recorrer, de principio a fin, un pipeline moderno de LLM Engineering:

```text
Modelo open-source
        ↓
Hugging Face
        ↓
Inferencia baseline
        ↓
Dataset especializado
        ↓
SFT
        ↓
LoRA
        ↓
Fine-tuning en GPU
        ↓
Evaluación
        ↓
Deployment
        ↓
API
        ↓
Demo
```

Y después utilizar el modelo especializado que construimos, **MentorLM**, para guiarnos en la construcción de un Transformer pequeño desde cero:

```text
Qwen3.5-4B
    ↓
Fine-tuning
    ↓
MentorLM
    ↓
"Enséñame a construir un GPT"
    ↓
TinyGPT
    ↓
Entrenamiento
    ↓
Modelo generativo pequeño
```

La historia completa del proyecto será:

> Tomamos un LLM open-source, lo especializamos como tutor técnico usando LoRA y GPUs serverless, evaluamos su mejora, lo desplegamos y posteriormente utilizamos ese mismo modelo para guiarnos en la implementación de un GPT pequeño desde primeros principios.

Esto conecta tres áreas:

```text
BUILD
+
FINE-TUNE
+
DEPLOY
```

---

# 2. Restricción de hardware local

La laptop disponible tiene aproximadamente:

```text
12 GB RAM
```

Por diseño, **no ejecutaremos cargas pesadas de ML localmente**.

La regla del proyecto será:

> **Laptop = desarrollo y orquestación. Modal = cómputo.**

La laptop se utilizará para:

- VS Code.
- Claude Code.
- Git.
- GitHub.
- Python ligero.
- Modal CLI.
- edición de archivos.
- documentación.
- terminal.
- navegador.

Modal se utilizará para:

- Linux remoto.
- contenedores.
- CPU y RAM remota.
- GPUs.
- CUDA.
- PyTorch.
- Transformers.
- Unsloth.
- carga del modelo.
- inferencia.
- entrenamiento.
- checkpoints.
- adapters LoRA.
- evaluación pesada.
- serving final.

Arquitectura conceptual:

```text
┌──────────────────────────────┐
│        LAPTOP 12 GB RAM      │
│                              │
│ VS Code                      │
│ Claude Code                  │
│ Git                          │
│ Modal CLI                    │
└──────────────┬───────────────┘
               │
               │ modal run
               ▼
┌──────────────────────────────┐
│            MODAL             │
│                              │
│ Container Linux              │
│ Python                       │
│ PyTorch                      │
│ CUDA                         │
│ Transformers                 │
│ Unsloth                      │
│ GPU                          │
│ RAM remota                   │
└──────────────┬───────────────┘
               │
        ┌──────┴────────────┐
        ▼                   ▼
 Hugging Face           Modal Volumes
                         models
                         datasets
                         checkpoints
                         adapters
```

No necesitamos descargar Qwen a la laptop.

---

# 3. Producto principal: MentorLM

## Concepto

MentorLM será un LLM especializado en **enseñar conceptos de inteligencia artificial, Transformers, entrenamiento y LLM Engineering en español**.

No intentaremos crear un modelo "más inteligente que Qwen".

Buscaremos modificar de manera medible su:

- estructura de respuesta;
- comportamiento pedagógico;
- consistencia;
- claridad;
- progresión conceptual;
- forma de explicar;
- capacidad para detectar errores conceptuales;
- capacidad para acompañar una implementación técnica paso a paso.

Ejemplo de comportamiento deseado:

```text
Usuario:
¿Qué es LoRA?

MentorLM:

1. Intuición
2. Definición técnica
3. Qué problema resuelve
4. Cómo funciona
5. Ejemplo
6. Error común
7. Comprueba tu comprensión
```

---

# 4. Hipótesis del proyecto

La hipótesis debe poder medirse.

## Hipótesis

> Un fine-tuning pequeño pero bien diseñado mediante SFT + LoRA puede hacer que un modelo generalista como Qwen3.5-4B siga de manera más consistente una metodología pedagógica especializada para enseñar LLM Engineering en español.

Esto significa que antes de entrenar debemos medir al modelo original.

Después repetiremos exactamente las mismas evaluaciones.

Nunca diremos "mejoró" solamente porque el loss bajó.

---

# 5. Modelo base

Modelo propuesto:

```text
Qwen3.5-4B
```

Para la primera versión utilizaremos la variante post-entrenada/chat, no la versión Base.

Razones:

- ya entiende instrucciones;
- ya conversa;
- permite establecer un baseline útil;
- es suficientemente pequeño para experimentar;
- podemos especializar comportamiento sin realizar full fine-tuning;
- encaja cómodamente en infraestructura de GPU accesible mediante Modal.

---

# 6. Estrategia de entrenamiento

## Lo que utilizaremos

```text
Supervised Fine-Tuning
+
LoRA
+
BF16
```

## Lo que NO utilizaremos inicialmente

```text
Full Fine-Tuning
QLoRA sobre Qwen3.5
RLHF
GRPO
multi-GPU
distributed training
```

La intención es comprender primero muy bien SFT + LoRA.

---

# 7. Qué es LoRA conceptualmente

En un modelo tradicional tendríamos matrices de pesos:

```text
W
```

En full fine-tuning actualizaríamos directamente grandes cantidades de esos parámetros.

LoRA mantiene los pesos originales congelados y aprende una actualización de bajo rango:

```text
W' = W + ΔW

ΔW = BA
```

Donde `A` y `B` son matrices mucho más pequeñas.

Visualmente:

```text
MODELO BASE

████████████████████████████
        congelado


LoRA adapter

██
entrenable
```

Esto permite adaptar un modelo con muchos menos parámetros entrenables.

Durante el proyecto será obligatorio comprender:

- `r`;
- `lora_alpha`;
- target modules;
- dropout;
- trainable parameters;
- frozen parameters;
- learning rate;
- batch size;
- gradient accumulation;
- epochs;
- max sequence length.

Ningún valor debe existir solamente porque Claude Code lo puso.

---

# 8. Infraestructura: Modal

Modal será nuestro laboratorio remoto.

Debemos aprender progresivamente:

```text
Modal App
Modal Image
Modal Function
Modal Container
Modal Volume
Modal Secret
GPU
CPU
RAM
VRAM
Remote execution
Cold start
Inference
Web endpoint
```

La primera meta será extremadamente sencilla:

```text
Laptop
   ↓
Modal CLI
   ↓
Remote Function
   ↓
GPU
   ↓
PyTorch
   ↓
torch.cuda.is_available() == True
```

Después:

```text
nvidia-smi
```

Cuando esto funcione, habremos validado nuestra infraestructura.

---

# 9. Hugging Face

Hugging Face será nuestro repositorio/ecosistema de modelos.

Durante esta fase debemos comprender:

- qué es un model repository;
- weights;
- `safetensors`;
- `config.json`;
- tokenizer;
- vocabulary;
- chat template;
- model card;
- Transformers;
- `AutoTokenizer`;
- carga del modelo;
- inferencia;
- generación;
- tokens;
- logits.

El modelo será descargado **desde Modal hacia Modal**, no a nuestra laptop.

```text
Hugging Face
      ↓
    Modal
      ↓
Model cache / Volume
```

---

# 10. Conceptos que debemos dominar

El proyecto será considerado exitoso solamente si al terminar podemos explicar, con nuestras propias palabras:

## Transformer

- tokens;
- embeddings;
- positional information;
- self-attention;
- Q, K y V;
- attention scores;
- causal mask;
- multi-head attention;
- MLP/feed-forward;
- residual connections;
- layer normalization;
- logits.

## Training

- forward pass;
- loss;
- cross entropy;
- backward pass;
- gradient;
- optimizer;
- AdamW;
- learning rate;
- batch;
- step;
- epoch;
- validation;
- overfitting;
- checkpoint.

## LLM Engineering

- pretraining;
- post-training;
- instruction tuning;
- SFT;
- PEFT;
- LoRA;
- quantization;
- BF16;
- VRAM;
- inference;
- serving;
- evaluation.

No necesitamos aprender todos antes de comenzar.

Los iremos entendiendo conforme aparezcan.

---

# 11. Dataset de MentorLM

No queremos generar miles de registros sin control.

Preferimos:

```text
300–600 ejemplos buenos
```

antes que:

```text
5,000 ejemplos mediocres
```

## Taxonomía inicial

```text
TRANSFORMERS
├── architecture
├── attention
├── self-attention
├── Q K V
├── embeddings
├── positional information
└── tokenization

TRAINING
├── forward pass
├── loss
├── cross entropy
├── gradient descent
├── backpropagation
├── optimizer
├── AdamW
├── learning rate
├── batch
├── epochs
└── checkpoints

FINE-TUNING
├── SFT
├── LoRA
├── PEFT
├── rank
├── adapters
├── full fine-tuning
└── quantization

INFERENCE
├── logits
├── sampling
├── temperature
├── top-p
├── context window
├── KV cache
└── serving

LLM ENGINEERING
├── evaluation
├── datasets
├── RAG
├── agents
├── deployment
├── GPUs
└── inference infrastructure
```

## Niveles

```text
beginner
intermediate
advanced
```

## Tipos de ejercicio

```text
explain
compare
analogy
technical explanation
debug misconception
implementation guidance
quiz
concept check
```

---

# 12. Formato del dataset

Ejemplo conceptual:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "¿Qué es LoRA?"
    },
    {
      "role": "assistant",
      "content": "..."
    }
  ]
}
```

Separación:

```text
data/
├── train.jsonl
├── validation.jsonl
└── test.jsonl
```

Regla:

> El test set nunca se utiliza durante entrenamiento.

---

# 13. Evaluación antes del entrenamiento

Antes del fine-tuning debemos construir un benchmark pequeño.

Meta inicial:

```text
50 prompts
```

Ninguno debe copiarse literalmente al training set.

Ejemplos:

```text
¿Qué es self-attention?

¿Por qué un Transformer necesita causal masking?

Explícame Q, K y V para alguien que apenas empieza.

¿Cuál es la diferencia entre una epoch y un training step?

¿LoRA modifica todos los pesos del modelo?

¿Qué diferencia existe entre inference y training?

¿Por qué existe la cuantización?

Explícame gradient accumulation y cuándo se usa.
```

## Rúbrica propuesta

Cada respuesta puede evaluarse de 0 a 2 en:

```text
Correctitud técnica
Intuición
Claridad
Estructura
Ejemplo
Pedagogía
Detección de errores conceptuales
```

Podremos crear una puntuación agregada.

---

# 14. Baseline

Ejecutaremos el benchmark contra:

```text
Qwen3.5-4B original
```

y guardaremos:

```text
evaluation/results/baseline.json
```

Posteriormente ejecutaremos exactamente el mismo benchmark contra MentorLM.

Entonces tendremos:

```text
BASE MODEL
     VS
FINE-TUNED MODEL
```

No inventaremos métricas.

Si el resultado real es:

```text
7.1 → 7.7
```

documentaremos exactamente eso.

Si una categoría empeora, también se documentará.

---

# 15. Experimentos de entrenamiento

Nunca entrenaremos directamente "la versión final".

## Experimento 001 — Smoke Test

Objetivo:

```text
¿Funciona el pipeline?
```

Debe validar:

- carga del modelo;
- carga del dataset;
- LoRA;
- trainer;
- algunos steps;
- loss;
- checkpoint;
- adapter guardado.

No busca calidad.

## Experimento 002 — Primera ejecución real

Objetivo:

- usar dataset completo;
- revisar learning curves;
- validar output;
- detectar problemas.

## Experimento 003 — Final

Solo se hará si identificamos algo concreto que mejorar.

Nunca cambiaremos cinco parámetros simultáneamente sin saber por qué.

---

# 16. Tracking de experimentos

Estructura:

```text
experiments/
├── 001-smoke-test/
├── 002-lora-r16/
└── 003-final/
```

Cada experimento debe registrar:

```yaml
model:
method:
dataset_version:
dataset_size:
rank:
lora_alpha:
learning_rate:
batch_size:
gradient_accumulation:
max_seq_length:
epochs:
train_loss:
eval_loss:
gpu:
runtime:
estimated_cost:
notes:
```

Esto convierte el proyecto en un experimento reproducible.

---

# 17. Deployment de MentorLM

Una vez validado el fine-tuning:

```text
LoRA adapter
     ↓
Modal Volume
     ↓
Inference container
     ↓
GPU
     ↓
Qwen + LoRA
     ↓
HTTP endpoint
```

Después:

```text
Web UI
   ↓
Modal Endpoint
   ↓
MentorLM
```

La UI final será mínima.

El protagonista es el experimento, no el frontend.

---

# 18. Demo principal

La interfaz ideal permitirá comparar:

```text
┌──────────────────────┬──────────────────────┐
│      QWEN BASE       │      MENTORLM        │
│                      │                      │
│ respuesta            │ respuesta            │
│                      │                      │
└──────────────────────┴──────────────────────┘
```

Esto permitirá enseñar visualmente qué cambió después del fine-tuning.

---

# 19. Segundo acto: MentorLM construye TinyGPT

Una vez desplegado MentorLM, lo utilizaremos como tutor.

Prompt general:

```text
Quiero construir un GPT pequeño desde cero en PyTorch.

No quiero que me des todo el código terminado.

Guíame componente por componente.

Antes de implementar cada pieza explícame:
1. qué problema resuelve;
2. cómo funciona;
3. qué forma tienen los tensores;
4. qué matemática necesitamos;
5. qué errores suelen cometerse;
6. cómo comprobamos que nuestra implementación funciona.
```

El flujo será:

```text
MentorLM
   ↓
Tokenizer / dataset
   ↓
Embeddings
   ↓
Positional information
   ↓
Causal self-attention
   ↓
Multi-head attention
   ↓
MLP
   ↓
Transformer block
   ↓
Language model
   ↓
Loss
   ↓
Backpropagation
   ↓
AdamW
   ↓
Training
   ↓
Generation
```

---

# 20. TinyGPT

TinyGPT NO necesita ser bueno.

Necesita ser comprensible.

Configuración aproximada:

```text
Vocabulary: pequeño
Context: 128–256
Layers: 4–6
Heads: 4
Embedding dimension: 256–384
Parameters: aproximadamente 10M–30M
```

Objetivo:

> Poder abrir `model.py` y comprender cada componente.

Ejemplo conceptual:

```python
self.token_embedding
self.position_embedding
self.attention
self.mlp
self.layer_norm
self.lm_head
```

Debemos poder responder:

- qué recibe;
- qué devuelve;
- shape de entrada;
- shape de salida;
- qué parámetros tiene;
- por qué existe.

TinyGPT también podrá entrenarse en Modal usando una GPU más económica.

---

# 21. TinyGPT como evaluación práctica de MentorLM

No solamente utilizaremos MentorLM.

Lo evaluaremos mientras nos enseña.

Por ejemplo:

```text
Prompt:
Explícame e implementa causal self-attention en PyTorch.
```

Lo enviamos a:

```text
Qwen3.5-4B original
```

y a:

```text
MentorLM
```

Comparamos:

- claridad;
- progresión pedagógica;
- precisión técnica;
- shapes;
- explicación matemática;
- código;
- errores comunes;
- ejercicios;
- capacidad para corregirnos.

TinyGPT se convierte así en un benchmark práctico adicional.

---

# 22. Registro de las sesiones MentorLM → TinyGPT

Guardaremos las sesiones:

```text
docs/tinygpt-sessions/
├── 01-tokenization.md
├── 02-embeddings.md
├── 03-self-attention.md
├── 04-multi-head-attention.md
├── 05-transformer-block.md
├── 06-training.md
└── 07-generation.md
```

Formato recomendado:

```text
Concepto
↓
Explicación de MentorLM
↓
Nuestra interpretación
↓
Implementación
↓
Prueba
↓
Errores
↓
Corrección
↓
Qué aprendimos
```

Esto hará que el repositorio también funcione como material educativo para otras personas.

---

# 23. Estructura propuesta del repositorio

```text
cloud-llm-lab/

├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
│
├── mentorlm/
│   ├── modal/
│   │   ├── 01_gpu_test.py
│   │   ├── 02_base_inference.py
│   │   ├── 03_train.py
│   │   └── 04_serve.py
│   │
│   ├── data/
│   │   ├── train.jsonl
│   │   ├── validation.jsonl
│   │   └── test.jsonl
│   │
│   ├── training/
│   ├── evaluation/
│   │   └── results/
│   └── experiments/
│
├── tinygpt/
│   ├── data/
│   ├── tokenizer.py
│   ├── model.py
│   ├── train.py
│   ├── generate.py
│   └── README.md
│
├── frontend/
│
└── docs/
    ├── architecture.md
    ├── learning-journal.md
    ├── modal.md
    ├── transformers.md
    ├── sft.md
    ├── lora.md
    ├── evaluation.md
    ├── lessons-learned.md
    └── tinygpt-sessions/
```

No es obligatorio crear toda la estructura el primer minuto.

Debe crecer conforme avancemos.

---

# 24. Git y GitHub

El repositorio debe existir desde el inicio.

Objetivo:

```text
Local project
   ↓
git init
   ↓
primer commit
   ↓
nuevo GitHub repository
   ↓
remote origin
   ↓
push
```

Idealmente utilizar:

```text
GitHub CLI
gh
```

si está instalado y autenticado.

Si no lo está:

1. verificar instalación;
2. instalarlo;
3. `gh auth login`;
4. crear repositorio;
5. vincular remote;
6. hacer push.

No se deben guardar:

- tokens;
- secrets;
- credenciales;
- API keys;
- archivos de modelo gigantes;
- caches.

---

# 25. Estrategia de commits

Queremos que Git cuente la historia del proyecto.

Ejemplos:

```text
chore: initialize cloud llm lab

feat(modal): add first remote function

feat(modal): verify gpu execution

feat(mentorlm): add qwen baseline inference

feat(data): add evaluation benchmark

feat(data): add training dataset v1

feat(training): add lora smoke test

feat(training): complete first mentorlm run

feat(eval): compare base and finetuned models

feat(serving): expose mentorlm endpoint

feat(tinygpt): implement token embeddings

feat(tinygpt): implement causal self attention
```

---

# 26. Notion

Notion funcionará como ML Learning Journal.

Página:

```text
Cloud LLM Lab
```

Secciones:

```text
Concepts Learned
Experiments
Questions
Errors & Solutions
Key Insights
Architecture Decisions
Costs
Next Experiments
```

Cada vez que aprendamos algo importante debe quedar escrito con nuestras propias palabras.

Ejemplo:

```text
LoRA no reemplaza al modelo base.

Mantiene congelados sus pesos y aprende matrices adicionales
de bajo rango que representan una actualización sobre determinadas
capas del modelo.
```

---

# 27. Rol de cada herramienta disponible

## Modal

Infraestructura ML:

- GPU;
- containers;
- training;
- inference;
- Volumes;
- endpoints.

## Hugging Face

- modelo;
- tokenizer;
- metadata;
- model repository;
- datasets si aplica.

## Claude Code Max

Pair programmer.

Debe:

- explicar;
- proponer;
- implementar;
- probar;
- documentar.

No debe convertirse en piloto automático.

## Gemini Ultra

Segundo crítico.

Útil especialmente para:

- auditar dataset;
- encontrar redundancia;
- detectar errores conceptuales;
- revisar calidad.

## GPT Plus

Tutor conceptual y segunda opinión.

Especialmente para:

- arquitectura;
- debugging conceptual;
- explicar ML;
- revisar experimentos.

## OpenCode

Uso posterior.

Posible experimento:

```text
MentorLM endpoint
       ↓
OpenCode
       ↓
agente utilizando nuestro modelo
```

No es camino crítico.

## Notion Business

Learning journal.

## Libro de Transformers/LLMs

Referencia teórica.

No copiar contenido protegido para convertirlo directamente en dataset.

Usarlo como:

```text
leer
↓
comprender
↓
tomar notas
↓
crear ejemplos originales
```

---

# 28. Plan de trabajo

## Fase 1 — Modal desde cero

### Objetivo

Ejecutar código Python remotamente y verificar acceso a GPU.

### Entregables

```text
mentorlm/modal/01_gpu_test.py
docs/modal.md
primeros commits
```

### Debemos aprender

- qué es Modal;
- local vs remote;
- App;
- Function;
- Image;
- container;
- GPU;
- VRAM.

### Definition of Done

```text
torch.cuda.is_available() == True
```

y `nvidia-smi` funcionando.

---

## Fase 2 — Inferencia del modelo original

### Objetivo

Ejecutar Qwen3.5-4B desde Modal.

### Entregables

```text
02_base_inference.py
baseline_examples.json
```

### Aprender

- Hugging Face;
- tokenizer;
- weights;
- inference;
- prompt/chat template;
- generation.

---

## Fase 3 — Benchmark

### Objetivo

Crear evaluación antes del entrenamiento.

### Entregables

```text
test.jsonl
baseline_results.json
evaluation rubric
```

---

## Fase 4 — Dataset

### Objetivo

Crear 300–600 ejemplos de calidad.

### Entregables

```text
train.jsonl
validation.jsonl
dataset audit
```

---

## Fase 5 — LoRA Smoke Test

### Objetivo

Validar pipeline de entrenamiento.

### Entregables

```text
03_train.py
experiment 001
checkpoint
adapter
```

---

## Fase 6 — Entrenamiento de MentorLM

### Objetivo

Realizar entrenamiento reproducible.

### Entregables

```text
MentorLM LoRA adapter
experiment config
training metrics
```

---

## Fase 7 — Evaluación

### Objetivo

Medir antes vs después.

### Entregables

```text
baseline.json
finetuned.json
comparison.json
examples.md
```

---

## Fase 8 — Serving

### Objetivo

Convertir MentorLM en servicio.

### Entregables

```text
04_serve.py
Modal endpoint
```

---

## Fase 9 — Demo

### Objetivo

Mostrar Base vs MentorLM.

### Entregables

```text
frontend/
demo
```

---

## Fase 10 — TinyGPT con MentorLM

### Objetivo

Utilizar MentorLM para enseñarnos a construir un GPT desde cero.

### Entregables

```text
tinygpt/
docs/tinygpt-sessions/
```

---

# 29. Qué NO vamos a hacer inicialmente

Para controlar scope:

```text
NO RAG
NO LangChain
NO vector database
NO MCP
NO agentes complejos
NO RLHF
NO GRPO
NO multi-GPU
NO Kubernetes
NO 70B
NO multimodal fine-tuning
NO full fine-tuning
```

OpenCode y otros extras solo entran cuando el pipeline principal esté terminado.

---

# 30. Definition of Done general

El proyecto principal está terminado cuando podamos marcar:

- [ ] Git configurado.
- [ ] Repositorio remoto creado.
- [ ] Modal CLI configurado.
- [ ] Primera función remota ejecutada.
- [ ] GPU remota verificada.
- [ ] Entiendo Modal App.
- [ ] Entiendo Modal Image.
- [ ] Entiendo Modal Function.
- [ ] Entiendo Modal Volume.
- [ ] Qwen cargado remotamente.
- [ ] Entiendo qué hace el tokenizer.
- [ ] Ejecuté inferencia baseline.
- [ ] Construí un benchmark.
- [ ] Construí dataset train/validation/test.
- [ ] Entiendo SFT.
- [ ] Entiendo LoRA.
- [ ] Entiendo rank.
- [ ] Sé qué parámetros son entrenables.
- [ ] Ejecuté smoke test.
- [ ] Ejecuté fine-tuning.
- [ ] Guardé adapter.
- [ ] Evalué datos no vistos.
- [ ] Comparé Base vs MentorLM.
- [ ] Desplegué MentorLM.
- [ ] Tengo una demo mínima.
- [ ] Documenté resultados.
- [ ] Documenté errores.
- [ ] Documenté costos.
- [ ] Puedo explicar la arquitectura sin ayuda.
- [ ] MentorLM fue utilizado para comenzar TinyGPT.

---

# 31. Filosofía de trabajo

## Regla 1 — Comprender antes de automatizar

Cada pieza importante debe explicarse antes de ocultarla detrás de una librería.

## Regla 2 — Un concepto por vez

Si aparece un término desconocido, se explica en contexto.

## Regla 3 — Experimentos pequeños primero

Antes de gastar GPU:

```text
smoke test
↓
validación
↓
ejecución real
```

## Regla 4 — Nada de magia de Claude

Claude puede programar.

Pero debe explicar:

```text
qué cambia
por qué
qué alternativa existe
cómo verificamos que funciona
```

## Regla 5 — Resultados honestos

No seleccionar solamente outputs bonitos.

También documentar:

- errores;
- respuestas malas;
- limitaciones;
- regresiones.

## Regla 6 — Reproducibilidad

Cada experimento debe poder reconstruirse desde Git y configuración.

---

# 32. Resultado final para compartir

El proyecto debería presentarse como:

# Cloud LLM Lab

**Build, fine-tune and deploy language models from first principles.**

```text
Qwen3.5-4B
+
Hugging Face
+
PyTorch
+
Unsloth
+
LoRA
+
Modal Serverless GPUs
+
Custom Dataset
+
Evaluation
+
Deployment
+
TinyGPT
```

Historia:

1. configuramos infraestructura serverless de GPU;
2. ejecutamos un modelo open-source;
3. construimos una evaluación;
4. creamos un dataset;
5. hicimos fine-tuning mediante LoRA;
6. medimos el resultado;
7. desplegamos el modelo;
8. usamos el propio modelo resultante para construir un Transformer pequeño.

---

# 33. Primer checkpoint

No debemos intentar hacer todo de golpe.

El primer checkpoint es exclusivamente:

```text
VS Code
↓
Git
↓
GitHub
↓
Modal CLI
↓
Modal remote function
↓
GPU
↓
PyTorch
↓
CUDA disponible
```

Una vez conseguido esto avanzaremos a Qwen.

---

# 34. Prompt inicial para Claude Code

Copia el siguiente prompt en una sesión nueva de Claude Code abierta dentro de la carpeta raíz del proyecto.

---

## PROMPT

Quiero que seas mi **pair programmer, tutor técnico y guía de ML Engineering** para el proyecto descrito en este documento.

Lee primero y analiza completamente el archivo:

`PROJECT_PLAN.md`

El proyecto se llama provisionalmente **Cloud LLM Lab** y el modelo principal que queremos construir será **MentorLM**.

### Mi contexto

Estoy aprendiendo.

Tengo fundamentos básicos de:

- LLMs;
- Transformers;
- attention;
- entrenamiento;
- conceptos generales de IA.

Pero nunca he utilizado:

- Modal;
- infraestructura serverless de GPU;
- Unsloth;
- LoRA en un proyecto real;
- Hugging Face para alojar/cargar modelos en este tipo de pipeline;
- serving de un modelo propio.

Mi laptop tiene solamente **12 GB de RAM**, por lo que NO quiero descargar ni ejecutar modelos pesados localmente.

La arquitectura debe seguir esta regla:

**Laptop = desarrollo/orquestación. Modal = cómputo pesado.**

Tengo una cuenta de Modal y actualmente solo tengo iniciada la sesión en su página web. No he configurado nada más.

También tengo disponibles:

- Claude Max 5x;
- Gemini Ultra;
- GPT Plus;
- OpenCode Go;
- Notion Business;
- GitHub;
- aproximadamente 280 USD de créditos en Modal;
- material/libros sobre Transformers y LLMs.

### Cómo quiero trabajar

NO quiero que construyas todo el proyecto automáticamente.

Quiero que me vayas guiando **paso a paso**.

Antes de cada bloque importante:

1. dime qué vamos a conseguir;
2. explícame el concepto involucrado;
3. dime qué se ejecutará localmente y qué se ejecutará en Modal;
4. dime qué archivos vamos a crear o modificar;
5. explícame los comandos antes de ejecutarlos;
6. implementa solamente el siguiente paso necesario;
7. verifica que haya funcionado;
8. explícame el resultado;
9. haz commit cuando tengamos un checkpoint lógico.

Si aparece un concepto como:

- container;
- Modal Image;
- Modal Function;
- Volume;
- GPU;
- VRAM;
- CUDA;
- tokenizer;
- logits;
- SFT;
- LoRA;
- rank;
- learning rate;
- gradient accumulation;
- batch size;
- BF16;
- checkpoint;

no asumas que lo domino.

Explícamelo en el momento en que aparezca y relaciónalo con lo que estamos construyendo.

### Git y GitHub

Quiero que el proyecto tenga control de versiones desde el inicio.

Primero:

1. revisa el estado actual de la carpeta;
2. verifica si Git está instalado;
3. inicia Git si todavía no existe repositorio;
4. crea un `.gitignore` apropiado para Python, Modal, modelos ML, caches y secretos;
5. crea un README inicial muy pequeño;
6. haz el primer commit.

Después verifica si `gh` (GitHub CLI) está instalado y autenticado.

Si está disponible:

- crea un repositorio NUEVO en mi cuenta;
- utiliza como nombre preferido `cloud-llm-lab`, si está disponible;
- enlázalo como `origin`;
- haz push del primer commit.

Si GitHub CLI no está instalado o no está autenticado:

- no inventes credenciales;
- explícame exactamente qué necesito hacer;
- guíame para instalar/autenticar `gh`;
- continúa cuando el entorno esté listo.

Nunca guardes:

- tokens;
- API keys;
- credenciales;
- secretos;
- caches gigantes;
- pesos completos de modelos

dentro del repositorio Git.

### Primera misión: aprender Modal desde cero

NO empieces todavía con Qwen.

NO empieces todavía con Hugging Face.

NO empieces todavía con Unsloth.

NO empieces todavía con LoRA.

Quiero primero entender y validar Modal.

Asume que lo único que tengo es:

- Windows;
- VS Code;
- una cuenta de Modal abierta en el navegador.

Quiero que me guíes desde literalmente cero.

Necesito que determines:

- si debo abrir VS Code;
- qué carpeta crear;
- qué terminal utilizar;
- si necesito Python;
- cómo verificar mi versión de Python;
- cómo instalar Modal CLI;
- cómo autenticar mi computadora con Modal;
- qué significa cada paso.

Después quiero crear el primer archivo del proyecto:

`mentorlm/modal/01_gpu_test.py`

Pero NO saltes directamente a una GPU.

Primero quiero entender:

1. cómo ejecutar una función local/remota sencilla;
2. qué es `modal.App`;
3. qué es una función remota;
4. qué ocurre cuando ejecuto `modal run`;
5. qué es un container.

Después avanzamos a GPU.

Quiero terminar la primera sesión consiguiendo:

```text
Laptop
↓
Modal CLI
↓
Modal remote function
↓
GPU remota
↓
PyTorch
↓
torch.cuda.is_available() == True
↓
nvidia-smi
```

Quiero ver claramente qué GPU nos asignó Modal y cuánta VRAM tiene.

Para esta primera prueba utiliza una GPU razonable y barata. No necesitamos H100 ni hardware excesivo.

### Muy importante sobre costos

Antes de cualquier operación que pueda gastar una cantidad significativa de créditos de Modal:

- explícame qué recurso vamos a solicitar;
- por qué;
- qué alternativa más barata existe;
- aproximadamente cuánto podría costar;
- cómo detener o evitar recursos innecesarios.

No quiero desperdiciar los 280 USD de crédito.

### Filosofía del proyecto

Quiero aprender, no solamente terminar.

No quiero escuchar:

"esto funciona porque la librería lo hace".

Quiero progresivamente comprender qué hay debajo.

Cuando más adelante lleguemos a MentorLM, quiero entender:

```text
modelo base
↓
tokenización
↓
dataset
↓
forward pass
↓
loss
↓
backpropagation
↓
LoRA
↓
optimizer
↓
checkpoint
↓
inference
```

Y cuando lleguemos a TinyGPT, quiero poder comprender cada componente del Transformer.

### Tu comportamiento

Actúa como un senior paciente trabajando conmigo.

Sé riguroso.

Si estoy entendiendo algo incorrectamente, corrígeme.

Si existe una decisión técnica importante, dame la razón.

No compliques la arquitectura innecesariamente.

No agregues RAG, agentes, LangChain, MCP, vector databases, Kubernetes ni tecnologías que no formen parte del objetivo actual.

No avances diez pasos por tu cuenta.

Trabajaremos mediante checkpoints pequeños y verificables.

### Ahora comienza

Primero:

1. lee `PROJECT_PLAN.md`;
2. inspecciona la carpeta donde estamos;
3. dame un resumen de máximo 10 puntos de lo que entiendes que vamos a construir;
4. dime cuál es nuestro **Checkpoint 1**;
5. revisa el entorno disponible;
6. inicia Git y prepara el repositorio;
7. vincúlalo con un repositorio nuevo de GitHub siguiendo las reglas anteriores;
8. después comienza conmigo desde cero la configuración de Modal.

No empieces Qwen, Unsloth ni LoRA todavía.

Nuestro objetivo inmediato es únicamente:

> **crear un entorno de trabajo reproducible y ejecutar exitosamente nuestro primer código remoto sobre una GPU de Modal, entendiendo cada paso.**

---

# 35. Cómo utilizar este prompt

1. Crea una carpeta local, por ejemplo:

```text
cloud-llm-lab
```

2. Guarda este documento como:

```text
PROJECT_PLAN.md
```

3. Abre esa carpeta en VS Code.

4. Abre Claude Code dentro de esa carpeta.

5. Pega únicamente el prompt anterior.

6. Sigue la sesión paso a paso.

7. No permitas que Claude salte directamente al entrenamiento.

El primer objetivo sigue siendo:

```text
Git
+
GitHub
+
Modal
+
Remote Function
+
GPU
```

Una vez logrado, el siguiente checkpoint será cargar Qwen3.5-4B desde Hugging Face directamente en Modal.

---

# 36. Principio final

El éxito del proyecto no será poder decir:

> Claude me entrenó un modelo.

Será poder decir:

> Entiendo cómo tomé un modelo open-source, lo ejecuté sobre infraestructura GPU serverless, construí un dataset, adapté su comportamiento con LoRA, medí el cambio, lo desplegué y posteriormente utilicé ese mismo modelo para ayudarme a construir un Transformer pequeño desde cero.

Ese es el estándar del proyecto.

