<template>
  <div id="app">
    <header class="app-header">
      <h1>PresentAgent</h1>
    </header>
    <main class="app-main">
      <router-view />
    </main>
  </div>
</template>

<script>
export default {
  name: 'App'
}
</script>

<style scoped>
#app {
  display: flex;
  flex-direction: column;
  height: 97vh; /* Make the app take the full viewport height */
  background-color: aliceblue;
  margin: 0;
}

.app-header {
  font-family: 'American Typewriter', serif;
  text-align: center;
  font-weight: bold;
  font-size: 20px;
  height: 10vh; /* Fixed height for header */
  color: saddlebrown;
  margin-top: 10vh;
  display: flex;
  align-items: center;
  justify-content: center;
}

.app-main {
  flex: 1; /* Main content takes up the remaining space */
  overflow:hidden; /* Prevent scrolling */
  font-family: "Arial Rounded MT Bold";
}

.app-footer {
  text-align: center;
  font-family: "Times New Roman", Times, serif;
  height: 10vh;
  display: flex;
  align-items: center;
  justify-content: center;
}
</style>
