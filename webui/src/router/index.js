import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '../views/HomeView.vue'
import LabelView from '../views/LabelView.vue'
import PipelineView from '../views/PipelineView.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      redirect: '/label',
    },
    {
      path: '/label',
      name: 'label',
      component: LabelView,
    },
    {
      path: '/pipeline',
      name: 'pipeline',
      component: PipelineView,
    },
  ],
})

export default router
