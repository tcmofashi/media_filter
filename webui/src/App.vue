<template>
  <div class="app-shell">
    <header class="top-nav">
      <RouterLink class="nav-item" to="/label">标注系统</RouterLink>
      <RouterLink class="nav-item" to="/pipeline">Pipeline</RouterLink>
    </header>
    <RouterView />
  </div>
</template>

<script setup>
import { RouterLink, RouterView } from 'vue-router'
</script>

<style scoped>
.app-shell {
  min-height: 100vh;
  padding: 1rem;
}

.top-nav {
  display: flex;
  gap: 0.8rem;
  margin-bottom: 1rem;
}

.nav-item {
  color: #e6edff;
  text-decoration: none;
  border: 1px solid rgba(255, 255, 255, 0.25);
  padding: 0.4rem 0.75rem;
  border-radius: 0.6rem;
}

.nav-item.router-link-active,
.nav-item.router-link-exact-active {
  border-color: #61dafb;
  box-shadow: 0 0 0 1px rgba(97, 218, 251, 0.55);
}

</style>
