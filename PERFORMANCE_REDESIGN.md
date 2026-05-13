# Plano de Redesign para Superar v1

## Diagnóstico

| Métrica | v1 | v2 | Diferença |
|---------|----|----|-----------|
| Tempo por step | ~36 µs | ~2100 µs | **59x mais lento** |
| Linguagem hot path | 100% Cython | 95% Python puro | Causa raiz |
| Objetos criados/step | ~0 | ~8 frozen dataclasses | Overhead |
| Cópias de array/step | 0 | 3-5 mask.copy() | Overhead |
| validate_invariants/step | 0 | 2 chamadas | Overhead |
| Chamadas numpy/step | centenas (vetorizado) | ~36.000 (micro-arrays) | **Causa raiz** |

### Onde o tempo é gasto no v2 (100% = 2100µs)

```
_build_analysis()                  48%  (~1008µs)  ← loop path×mod com numpy micro-calls
  ├─ _analyze_free_mask()          25%  (~525µs)   ← np.pad, np.diff, np.flatnonzero em 24 elems
  ├─ _summary_after_allocation()   21%  (~441µs)   ← chamado N vezes por candidato
  └─ candidate_starts_array()       2%  (~42µs)    ← Cython, rápido
_build_path_slot_features()        30%  (~630µs)   ← 9 features × k_paths × slots
_build_path_mod_features()          8%  (~168µs)
_build_path_features()              5%  (~105µs)
_build_link_metrics()               4%  (~84µs)    ← loop por link com numpy
Overhead (dataclasses, copies)      5%  (~105µs)
```

**Conclusão**: O v2 é lento porque faz dezenas de milhares de chamadas Python→numpy em arrays minúsculos (24 slots). Cada `np.flatnonzero()`, `np.pad()`, `np.clip()`, `np.diff()` em 24 elementos custa ~3-8µs de overhead Python, mas a operação C é <0.1µs.

---

## Estratégia: Cythonizar o Hot Path

O v1 é rápido porque TUDO está em Cython. O v2 tem boa arquitetura modular mas executou o hot path em Python puro. A solução não é reescrever tudo no estilo v1 monolítico, mas mover o que importa para Cython mantendo a organização.

### Princípios

1. **Cython onde importa**: `request_analysis.py` → `request_analysis.pyx`
2. **Python onde não importa**: configs, topology loading, traffic table I/O ficam Python
3. **Eliminar overhead per-step**: frozen dataclasses, validações, cópias
4. **Eliminar numpy micro-calls**: C loops diretos em memoryviews
5. **Observação lazy**: não computar features dentro do step, só quando pedido

---

## Fase 1: Quick Wins (sem mudança de arquitetura)

Estes ganhos são imediatos e não requerem Cython. Estimativa: **reduzir de 2100µs para ~1600µs**.

### 1.1 Remover `validate_invariants()` do hot path

**Arquivo**: `simulation/simulator.py` linha ~655
```python
# REMOVER esta linha de _apply_action():
self.state.validate_invariants()
```

**Arquivo**: `stats/statistics.py` método `record_transition()`
```python
# REMOVER a chamada validate_invariants() de record_transition()
```

Manter apenas em `reset()` ou atrás de flag `debug=True`.

### 1.2 Eliminar cópias redundantes de mask

No `step()` do simulator.py:
- Linha 139: `current_mask = self.current_mask.copy()` → necessária (mask vai para StepTransition)
- Linha 167: `next_mask = self.current_mask.copy()` → **DESNECESSÁRIA**, já é uma cópia interna  
- Linha 229: `info["mask"] = next_mask.copy()` → **SEGUNDA CÓPIA** desnecessária
- Linha 252: `self.current_snapshot.flat.copy()` → necessária para gym

**Ação**: Eliminar pelo menos 2 cópias. Usar referência direta onde possível.

### 1.3 Simplificar `StepTransition`

Trocar `@dataclass(frozen=True, slots=True)` por `@dataclass(slots=True)` (mutable).
Remover `__post_init__` validation (valida `mask.ndim`, `disrupted_services >= 0` toda chamada).

### 1.4 Simplificar `Allocation`

O `__post_init__` valida 6 campos opcionais baseado em accepted/rejected. Custa ~5µs por step.
Trocar por factory methods `Allocation.accept()` e `Allocation.reject()` que já existem, e remover a validação.

### 1.5 Simplificar `Statistics.snapshot()`

Criar `StatisticsSnapshot` (17 campos frozen) todo step é caro.
Expor estatísticas diretamente do objeto `Statistics` sem criar snapshot intermediário.

### 1.6 Simplificar `StepInfo.build()`

Criar dict com ~25 chaves todo step. A maioria nem é lida pelo agente RL.
Fazer lazy: só construir quando `info` é acessado, ou ter um modo "minimal" para treinamento.

### 1.7 Simplificar `RewardFunction.evaluate()`

Retornar `(float, dict)` direto em vez de criar `RewardBreakdown` frozen dataclass toda vez.
Ou tornar `RewardBreakdown` mutable.

---

## Fase 2: Cythonizar `request_analysis.py` (o salto grande)

Esta é a mudança que vai fazer o v2 superar o v1. Estimativa: **reduzir de ~1600µs para ~30-50µs**.

### 2.1 Converter `request_analysis.py` → `request_analysis.pyx`

**Renomear** o arquivo e adicionar ao `setup.py`:

```python
# setup.py - adicionar
Extension(
    "optical_networking_gym_v2.simulation.request_analysis",
    ["src/optical_networking_gym_v2/simulation/request_analysis.pyx"],
    include_dirs=[np.get_include()],
)
```

### 2.2 Reescrever `_analyze_free_mask()` com C loops

O original usa np.pad, np.diff, np.flatnonzero em 24 elementos. Em Cython:

```cython
cimport numpy as cnp
from libc.math cimport log, sqrt

cdef struct FreeRunAnalysis:
    int count
    int largest
    int total_free
    double entropy
    double rss
    double sum_squares
    double sum_length_log_length

cdef FreeRunAnalysis analyze_free_mask_c(cnp.int8_t[:] free_mask, int total_slots) noexcept nogil:
    cdef FreeRunAnalysis result
    cdef int i, run_len, start
    cdef bint in_run = False
    cdef int max_runs = (total_slots + 1) // 2
    
    result.count = 0
    result.largest = 0
    result.total_free = 0
    result.entropy = 0.0
    result.rss = 0.0
    result.sum_squares = 0.0
    result.sum_length_log_length = 0.0
    
    run_len = 0
    for i in range(total_slots):
        if free_mask[i]:
            run_len += 1
        else:
            if run_len > 0:
                result.count += 1
                if run_len > result.largest:
                    result.largest = run_len
                result.total_free += run_len
                result.sum_squares += <double>(run_len * run_len)
                if run_len > 0:
                    result.sum_length_log_length += run_len * log(<double>run_len)
                run_len = 0
    if run_len > 0:
        result.count += 1
        if run_len > result.largest:
            result.largest = run_len
        result.total_free += run_len
        result.sum_squares += <double>(run_len * run_len)
        if run_len > 0:
            result.sum_length_log_length += run_len * log(<double>run_len)
    
    if result.total_free > 0 and result.count > 1:
        result.entropy = (log(<double>result.total_free) - result.sum_length_log_length / result.total_free) / log(<double>result.count)
        if result.entropy < 0.0:
            result.entropy = 0.0
        if result.entropy > 1.0:
            result.entropy = 1.0
    
    if result.total_free > 0:
        result.rss = sqrt(result.sum_squares) / result.total_free
        if result.rss > 1.0:
            result.rss = 1.0
    
    return result
```

**Ganho esperado**: `_analyze_free_mask()` de ~525µs → ~1µs (500x).

### 2.3 Reescrever `_build_link_metrics()` com C loops

```cython
cdef void build_link_metrics_c(
    cnp.int32_t[:, :] slot_allocation,
    cnp.float32_t[:, :] metrics_out,
    int link_count,
    int total_slots,
) noexcept nogil:
    cdef int link_id, slot, occupied, first_used, last_used, span_width
    cdef double compactness
    cdef FreeRunAnalysis analysis
    cdef cnp.int8_t free_mask_buf[512]  # max slots assumido
    cdef int max_block_count = (total_slots + 1) // 2
    
    for link_id in range(link_count):
        occupied = 0
        first_used = -1
        last_used = -1
        for slot in range(total_slots):
            if slot_allocation[link_id, slot] == -1:
                free_mask_buf[slot] = 1
            else:
                free_mask_buf[slot] = 0
                occupied += 1
                if first_used == -1:
                    first_used = slot
                last_used = slot
        
        analysis = analyze_free_mask_c(free_mask_buf[:total_slots], total_slots)
        
        if occupied == 0 or occupied == total_slots:
            compactness = 1.0
        else:
            span_width = last_used - first_used + 1
            compactness = <double>occupied / <double>span_width if span_width > 0 else 1.0
        
        metrics_out[link_id, 0] = <float>(occupied) / <float>(total_slots)
        metrics_out[link_id, 1] = <float>analysis.entropy
        metrics_out[link_id, 2] = <float>(1.0 - <double>analysis.largest / <double>analysis.total_free) if analysis.total_free > 0 else 0.0
        metrics_out[link_id, 3] = <float>compactness
        metrics_out[link_id, 4] = <float>analysis.count / <float>max_block_count
        metrics_out[link_id, 5] = <float>analysis.rss
```

**Ganho**: `_build_link_metrics()` de ~84µs → <2µs.

### 2.4 Reescrever `_build_path_slot_features()` com C loops

Esta função é 30% do tempo total. O original faz:
- `_analyze_free_mask()` (de novo, para cada path)
- `_slot_block_vectors()` com `_fill_slot_run_vectors()` (arrays numpy)
- Loop por slot fazendo `np.flatnonzero()` em cada um
- `_local_fragmentation()` com `np.convolve()` em 24 elementos

Em Cython, tudo isso vira loops C diretos:

```cython
cdef void build_path_slot_features_c(
    cnp.int8_t[:, :] common_free_masks,      # (k_paths, total_slots)
    cnp.uint8_t[:, :, :] resource_valid,      # (k_paths, mods, total_slots)
    cnp.uint8_t[:, :, :] qot_valid,           # (k_paths, mods, total_slots)
    cnp.float32_t[:, :, :] osnr_margin,       # (k_paths, mods, total_slots)
    cnp.float32_t[:, :, :] nli_share,         # (k_paths, mods, total_slots)
    cnp.float32_t[:, :] result,               # (k_paths * total_slots, 9)
    int k_paths, int mod_count, int total_slots, int window,
) noexcept nogil:
    # Implementação com loops C diretos
    # Sem np.flatnonzero, sem np.convolve, sem np.pad
    pass
```

**Ganho**: `_build_path_slot_features()` de ~630µs → ~5µs.

### 2.5 Reescrever `_summary_after_allocation()` como C function

O original é chamado N vezes por path×mod (para cada candidato). Já recebe `_FreeRunAnalysis` pre-computada, mas usa `np.clip()` e `np.sqrt()` Python.

```cython
cdef void summary_after_allocation_c(
    FreeRunAnalysis* free_runs,
    int* run_starts,
    int* run_ends,
    int* run_lengths,
    int* slot_to_run_index,
    int* largest_other_by_run,
    int service_slot_start,
    int service_num_slots,
    int total_slots,
    int* post_count,
    int* post_largest,
) noexcept nogil:
    # Cálculo direto em C sem overhead Python
    pass
```

**Ganho**: `_summary_after_allocation()` de ~441µs → <3µs.

### 2.6 Reescrever `_build_analysis()` como cpdef

O loop principal path×modulation com candidate_starts e fragmentation damage vira Cython.
A parte de OSNR (call para `qot_engine.summarize_candidate_starts`) já é Cython e fica.

```cython
cpdef RequestAnalysis build_analysis(
    object config,
    object topology,
    object qot_engine,
    object state,
    object request,
    dict path_link_indices,
):
    # Principal loop em Cython com typed variables
    # Chama analyze_free_mask_c, summary_after_allocation_c direto
    pass
```

---

## Fase 3: Eliminar Over-Engineering

### 3.1 Contracts → Plain classes ou NamedTuples

| Contract | Campos | `__post_init__` | Criações/step | Ação |
|----------|--------|-----------------|---------------|------|
| `ServiceRequest` | 11 | Valida todos | 1 | Remover __post_init__ |
| `Allocation` | 8 | Valida 6 por status | 1 | Remover __post_init__ |
| `StepTransition` | 11 | Valida mask, disrupted | 1 | `@dataclass(slots=True)` mutable |
| `StatisticsSnapshot` | 17 | Nenhum | 1 (via snapshot()) | **Eliminar**, expor Statistics direto |
| `RewardBreakdown` | 8 | Nenhum | 1 | `@dataclass(slots=True)` mutable |
| `ObservationSnapshot` | 8 | Nenhum | 1 | **Eliminar**, usar tuple direto |
| `CandidateRewardMetrics` | 5 | Nenhum | 1 | Tuple ou struct |

**Regra**: Nenhum `frozen=True` no hot path. Nenhum `__post_init__` com validação no hot path.

### 3.2 Eliminar `ObservationSnapshot`

O `Observation.build_snapshot()` cria um `ObservationSnapshot` que empacota flat array + analysis + schema. Isso é overhead.

```python
# ANTES (observation.py):
def build_snapshot(self, state, request):
    analysis = self.analysis_engine.build(state, request)
    flat = np.concatenate([
        analysis.request_features,
        analysis.global_features,
        analysis.path_features.ravel(),
        analysis.path_mod_features.ravel(),
        analysis.path_slot_features.ravel(),
    ])
    return ObservationSnapshot(flat=flat, analysis=analysis, ...)

# DEPOIS: retornar (flat, analysis) direto, sem snapshot
```

### 3.3 Eliminar `StatisticsSnapshot`

```python
# ANTES: toda step cria StatisticsSnapshot com 17 campos frozen
snapshot = self.statistics.snapshot()
reward_input = RewardInput(statistics=snapshot, ...)

# DEPOIS: reward_function recebe Statistics direto
reward_value = self.reward_function.evaluate(transition, self.statistics, analysis)
```

### 3.4 Eliminar `RewardInput`

```python
# ANTES: cria RewardInput frozen dataclass empacotando tudo
reward_input = RewardInput(
    transition=transition,
    statistics=snapshot,
    request_analysis=analysis,
    has_valid_non_reject_action=...,
)

# DEPOIS: passar argumentos direto
```

### 3.5 Simplificar `StepInfo.build()`

```python
# ANTES: dict com 25+ chaves criado toda step
# DEPOIS: modo "training" que retorna dict minimal
def build_training(self, statistics, mask):
    return {"mask": mask}
```

---

## Fase 4: Observação Lazy

### Problema atual

No v2, `_prepare_next_request()` é chamado DENTRO de `step()`:
```
step(action):
    _apply_action(action)           # executa ação
    _prepare_next_request()         # gera próximo request E computa observação!
    return observation              # observação já estava pronta
```

O v1 faz diferente:
```
step(action):
    _process_action(action)         # executa ação
    _next_service()                 # gera próximo request (sem observação)
    obs = observation_optimized()   # observação separada
    return obs
```

Na prática, ambos computam a observação no step. Mas o v2 obriga a computar tudo junto por causa da arquitetura analysis→snapshot.

### Solução

Manter o flow como está (observação no step), mas garantir que `_build_analysis()` em Cython seja tão rápido quanto `observation_optimized()` em v1. Com Fase 2 implementada, isso acontece automaticamente.

---

## Fase 5: Ajustes no `setup.py`

```python
# Adicionar ao setup.py do v2:
Extension(
    "optical_networking_gym_v2.simulation.request_analysis",
    ["src/optical_networking_gym_v2/simulation/request_analysis.pyx"],
    include_dirs=[np.get_include()],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
),
```

Compiler directives (já existentes, manter):
```python
compiler_directives={
    "boundscheck": False,
    "wraparound": False,
    "initializedcheck": False,
}
```

---

## Ordem de Implementação Recomendada

| # | Tarefa | Impacto | Risco |
|---|--------|---------|-------|
| 1 | Remover validate_invariants do hot path | -50µs | Zero |
| 2 | Remover cópias redundantes de mask | -30µs | Baixo |
| 3 | Simplificar contracts (remover frozen/validation) | -30µs | Baixo |
| 4 | Eliminar StatisticsSnapshot, RewardInput, ObservationSnapshot | -40µs | Médio |
| 5 | `_analyze_free_mask()` → Cython | -500µs | Médio |
| 6 | `_build_link_metrics()` → Cython | -80µs | Baixo |
| 7 | `_build_path_slot_features()` → Cython | -600µs | Alto |
| 8 | `_summary_after_allocation()` → Cython | -400µs | Médio |
| 9 | `_build_analysis()` loop principal → Cython | -200µs | Alto |
| 10 | `_build_path_mod_features()` → Cython | -160µs | Médio |
| 11 | StepInfo minimal mode para training | -10µs | Zero |

**Meta**: Após todas as fases, step time < 40µs (igual ou melhor que v1).

---

## O que NÃO mudar

1. **Topology loading** (`TopologyModel.from_file()`): executa uma vez, não importa
2. **Traffic model** (`TrafficModel`): já é eficiente
3. **QoT kernels** (`qot_kernel.pyx`, `allocation_kernel.pyx`): já são Cython otimizado
4. **Organização de diretórios**: manter a estrutura modular
5. **Interfaces Python** (`gymnasium wrapper`, `ScenarioConfig`): não estão no hot path
6. **Step trace / capture**: só executa quando `capture_step_trace=True`

---

## Resumo

O v2 é 59x mais lento que v1 não por causa da arquitetura modular, mas porque o hot path INTEIRO (`request_analysis.py`) foi implementado em Python puro com milhares de micro-chamadas numpy.

A solução é cirúrgica: converter `request_analysis.py` para `.pyx` com loops C diretos, eliminar overhead de objetos no step loop, e manter a organização modular para tudo que não é hot path.

O v1 prova que Cython neste domínio é ~50-100x mais rápido que Python+numpy para operações em arrays pequenos. O v2 tem arquitetura melhor; falta apenas compilar o código certo.
