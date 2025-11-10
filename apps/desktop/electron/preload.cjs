const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('notebookBridge', {
  ping: () => ipcRenderer.invoke('app:ping'),
  choosePath: (options) => ipcRenderer.invoke('dialog:choosePath', options),
  openExternal: (url) => ipcRenderer.invoke('app:openExternal', url),
});

