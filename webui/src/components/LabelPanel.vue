<template>
  <div class="label-panel">
    <div v-if="!selectedFile" class="no-selection">
      <p>选择一个媒体文件以开始标注</p>
    </div>
    
    <div v-else class="panel-content">
      <div class="file-preview">
        <span class="file-label">已选择: {{ selectedFile.name }}</span>
        <span v-if="selectedFile.inference_score != null" class="inference-score">
          模型分: {{ formatScore(selectedFile.inference_score) }}
        </span>
        <span v-if="existingLabel" class="existing-score">
          当前评分: {{ existingLabel.score }}
        </span>
      </div>
      
      <!-- Score Buttons -->
      <div class="score-section">
        <label>评分 (0最低, 9最高)</label>
        <div class="score-buttons">
          <button
            v-for="n in 10"
            :key="n-1"
            class="score-btn"
            :class="{ active: score === n-1 }"
            @click="quickScore(n-1)"
          >
            {{ n-1 }}
          </button>
        </div>
      </div>
      
      <!-- Navigation -->
      <div class="actions">
        <button
          @click="goToPrevious"
          class="nav-btn"
        >
          ← 上一个
        </button>
        
        <button
          @click="goToNext"
          class="nav-btn"
        >
          下一个 →
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted, onUnmounted, toRaw } from 'vue'

const props = defineProps({
  selectedFile: {
    type: Object,
    default: null
  },
  labels: {
    type: Array,
    default: () => []
  },
  isPreviewOpen: {
    type: Boolean,
    default: false
  }
})

const emit = defineEmits(['save', 'next', 'previous'])

const score = ref(5)
const isSaving = ref(false)

function toPresetScore(scoreValue) {
  const numericScore = Number(scoreValue)
  if (!Number.isFinite(numericScore)) return null
  return Math.max(0, Math.min(9, Math.round(numericScore)))
}

const existingLabel = computed(() => {
  if (!props.selectedFile) return null
  const rawLabels = toRaw(props.labels)
  if (rawLabels) {
    for (let i = rawLabels.length - 1; i >= 0; i--) {
      if (rawLabels[i].media_path === props.selectedFile.path) {
        return rawLabels[i]
      }
    }
  }
  return null
})

const matchedPresetScore = computed(() => {
  const labeledScore = toPresetScore(existingLabel.value?.score)
  if (labeledScore !== null) return labeledScore

  const inferenceScore = toPresetScore(props.selectedFile?.inference_score)
  if (inferenceScore !== null) return inferenceScore

  return 5
})

watch(matchedPresetScore, (value) => {
  score.value = value
}, { immediate: true })

async function saveLabel() {
  if (!props.selectedFile || isSaving.value) return
  
  isSaving.value = true
  try {
    emit('save', {
      mediaPath: props.selectedFile.path,
      score: score.value
    })
  } finally {
    isSaving.value = false
  }
}

// Quick score: save immediately; parent view advances selection.
function quickScore(newScore) {
  score.value = newScore
  saveLabel()
}

function goToNext() {
  emit('next')
}

function goToPrevious() {
  emit('previous')
}

function formatScore(scoreValue) {
  return Number(scoreValue).toFixed(4).replace(/\.?0+$/, '')
}

function handleKeydown(event) {
  if (props.isPreviewOpen) return

  // Number keys 0-9 for quick scoring.
  if (event.key >= '0' && event.key <= '9') {
    quickScore(parseInt(event.key))
    event.preventDefault()
    return
  }
  if (event.key === 'ArrowRight') {
    goToNext()
    event.preventDefault()
  }
  if (event.key === 'ArrowLeft') {
    goToPrevious()
    event.preventDefault()
  }
}

onMounted(() => {
  window.addEventListener('keydown', handleKeydown)
})

onUnmounted(() => {
  window.removeEventListener('keydown', handleKeydown)
})

defineExpose({ score })
</script>

<style scoped>
.label-panel {
  background: #1a1a1a;
  border-top: 1px solid #333;
  padding: 16px 20px;
}

.no-selection {
  text-align: center;
  color: #666;
  padding: 20px;
  font-size: 14px;
}

.panel-content {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.file-preview {
  display: flex;
  align-items: center;
  gap: 16px;
  padding-bottom: 12px;
  border-bottom: 1px solid #333;
}

.file-label {
  font-size: 14px;
  color: #ccc;
}

.existing-score {
  font-size: 13px;
  color: #42b883;
  background: rgba(66, 184, 131, 0.1);
  padding: 4px 10px;
  border-radius: 12px;
}

.inference-score {
  font-size: 13px;
  color: #7ad7aa;
  background: rgba(122, 215, 170, 0.1);
  padding: 4px 10px;
  border-radius: 12px;
}

.score-section {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.score-section label {
  font-size: 12px;
  color: #888;
  font-weight: 500;
}

.score-buttons {
  display: flex;
  gap: 8px;
  justify-content: center;
}

.score-btn {
  width: 36px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #2a2a2a;
  border: 1px solid #444;
  border-radius: 6px;
  color: #ccc;
  font-family: 'Orbitron', sans-serif;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.1s ease;
}

.score-btn:hover {
  background: rgba(66, 184, 131, 0.2);
  border-color: rgba(66, 184, 131, 0.5);
  color: #42b883;
}

.score-btn.active {
  background: #42b883;
  border-color: #42b883;
  color: #0a0a0a;
}

.actions {
  display: flex;
  gap: 12px;
  margin-top: 4px;
}

.nav-btn {
  flex: 1;
  padding: 10px 16px;
  background: #2a2a2a;
  border: 1px solid #444;
  border-radius: 6px;
  color: #ccc;
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.08s;
}

.nav-btn:hover {
  background: #333;
  border-color: #555;
  color: #fff;
}
</style>
