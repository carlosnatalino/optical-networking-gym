#!/bin/bash
# Script para iniciar treinamento LOW_PB_STABLE em screen
# Uso: ./start_training.sh

set -e

# Cores para output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Iniciando Treinamento LOW_PB_STABLE${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Configurações
SESSION_NAME="train_lowpb_stable"
SCRIPT_PATH="/home/talles/projects/optical-networking-gym/examples/OFC_2025/train_n_steps.py"
VENV_PYTHON="/home/talles/projects/optical-networking-gym/.venv/bin/python"
WORK_DIR="/home/talles/projects/optical-networking-gym"

# Verifica se já existe uma sessão
if screen -list | grep -q "$SESSION_NAME"; then
    echo -e "${YELLOW}⚠️  Sessão '$SESSION_NAME' já existe!${NC}"
    echo ""
    echo "Opções:"
    echo "  1. Ver sessão existente:  screen -r $SESSION_NAME"
    echo "  2. Matar e criar nova:    screen -X -S $SESSION_NAME quit && $0"
    echo ""
    exit 1
fi

# Verifica se Python e script existem
if [ ! -f "$VENV_PYTHON" ]; then
    echo -e "${YELLOW}❌ Python não encontrado: $VENV_PYTHON${NC}"
    exit 1
fi

if [ ! -f "$SCRIPT_PATH" ]; then
    echo -e "${YELLOW}❌ Script não encontrado: $SCRIPT_PATH${NC}"
    exit 1
fi

# Cria diretório de logs
LOG_DIR="/home/talles/projects/optical-networking-gym/examples/OFC_2025/training_runs"
mkdir -p "$LOG_DIR"

# Timestamp para o log
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/train_lowpb_${TIMESTAMP}.log"

echo -e "${GREEN}✓${NC} Configurações:"
echo "   Session: $SESSION_NAME"
echo "   Script:  $SCRIPT_PATH"
echo "   Python:  $VENV_PYTHON"
echo "   WorkDir: $WORK_DIR"
echo "   LogFile: $LOG_FILE"
echo ""

# Cria a sessão screen e inicia o treinamento
echo -e "${BLUE}🚀 Iniciando sessão screen...${NC}"
screen -dmS "$SESSION_NAME" bash -c "
    cd '$WORK_DIR' && \
    echo '============================================' && \
    echo '  Treinamento LOW_PB_STABLE Iniciado' && \
    echo '  $(date)' && \
    echo '============================================' && \
    echo '' && \
    '$VENV_PYTHON' '$SCRIPT_PATH' 2>&1 | tee '$LOG_FILE' && \
    echo '' && \
    echo '============================================' && \
    echo '  Treinamento Concluído!' && \
    echo '  $(date)' && \
    echo '============================================' && \
    echo '' && \
    echo 'Pressione qualquer tecla para fechar...' && \
    read -n 1
"

# Aguarda um pouco para garantir que iniciou
sleep 2

# Verifica se a sessão foi criada
if screen -list | grep -q "$SESSION_NAME"; then
    echo -e "${GREEN}✓${NC} Sessão criada com sucesso!"
    echo ""
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}  Treinamento em Execução!${NC}"
    echo -e "${BLUE}============================================${NC}"
    echo ""
    echo "📊 Para acompanhar o treinamento:"
    echo ""
    echo "   1. Conectar na sessão:"
    echo "      ${GREEN}screen -r $SESSION_NAME${NC}"
    echo ""
    echo "   2. Desconectar (sem parar):"
    echo "      Pressione: ${GREEN}Ctrl+A, depois D${NC}"
    echo ""
    echo "   3. Ver log em tempo real:"
    echo "      ${GREEN}tail -f $LOG_FILE${NC}"
    echo ""
    echo "   4. Monitorar com TensorBoard:"
    echo "      ${GREEN}tensorboard --logdir $LOG_DIR${NC}"
    echo ""
    echo "   5. Listar sessões screen:"
    echo "      ${GREEN}screen -list${NC}"
    echo ""
    echo "   6. Matar sessão (se necessário):"
    echo "      ${GREEN}screen -X -S $SESSION_NAME quit${NC}"
    echo ""
    echo -e "${YELLOW}⏱️  Tempo estimado: ~20-30 horas${NC}"
    echo -e "${YELLOW}📈 Meta: Blocking Rate < 1%${NC}"
    echo ""
else
    echo -e "${YELLOW}❌ Erro ao criar sessão!${NC}"
    exit 1
fi
