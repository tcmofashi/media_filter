const fs = require('fs');
let code = fs.readFileSync('src/components/MediaPreview.vue', 'utf8');

if (!code.includes('toRaw')) {
  code = code.replace(/import { ref, computed, watch, onMounted, onUnmounted } from 'vue'/, "import { ref, computed, watch, onMounted, onUnmounted, toRaw } from 'vue'");
}

code = code.replace(/const currentScore = computed\(\(\) => \{[\s\S]*?return label \? label\.score : 5\n\}\)/, `const currentScore = computed(() => {
  if (!currentFile.value) return 5
  
  const rawLabels = toRaw(props.labels)
  // reverse search since newest labels are generally at the end
  for (let i = rawLabels.length - 1; i >= 0; i--) {
    if (rawLabels[i].media_path === currentFile.value.path) {
      return rawLabels[i].score
    }
  }
  return 5
})`);

fs.writeFileSync('src/components/MediaPreview.vue', code);
