// assets/sw.js
self.addEventListener('install', (event) => {
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(self.clients.claim());
});

// (Optional) Basic fetch passthrough; customize if you want caching.
self.addEventListener('fetch', () => {});
