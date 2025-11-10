const { app, BrowserWindow, ipcMain, dialog, shell } = require('electron');
const path = require('node:path');
const { spawn } = require('node:child_process');

const isDev = process.env.NODE_ENV === 'development';
const VITE_DEV_SERVER_URL = process.env.VITE_DEV_SERVER_URL || 'http://localhost:5173';

let backendProcess = null;

function createMainWindow() {
  const mainWindow = new BrowserWindow({
    width: 1280,
    height: 800,
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  if (isDev) {
    mainWindow.loadURL(VITE_DEV_SERVER_URL);
    // DevTools can be opened manually with Cmd+Option+I (Mac) or Ctrl+Shift+I (Windows/Linux)
    // mainWindow.webContents.openDevTools({ mode: 'detach' });
  } else {
    mainWindow.loadFile(path.join(__dirname, '..', 'dist', 'index.html'));
  }

  return mainWindow;
}

function startBackend() {
  if (backendProcess) {
    return;
  }

  const python = process.env.NOTEBOOKLM_PYTHON || 'uv';
  const useUv = python === 'uv';

  if (useUv) {
    backendProcess = spawn('uv', ['run', 'uvicorn', 'notebooklm_backend.app:create_app', '--factory', '--host', '127.0.0.1', '--port', '8000'], {
      cwd: path.join(__dirname, '..', '..', 'backend'),
      stdio: 'inherit',
    });
  } else {
    backendProcess = spawn(python, ['-m', 'uvicorn', 'notebooklm_backend.app:create_app', '--factory', '--host', '127.0.0.1', '--port', '8000'], {
      cwd: path.join(__dirname, '..', '..', 'backend'),
      stdio: 'inherit',
    });
  }

  backendProcess.on('exit', (code) => {
    backendProcess = null;
    console.log(`[backend] exited with code ${code}`);
  });
}

app.whenReady().then(() => {
  if (!isDev) {
    startBackend();
  }

  createMainWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createMainWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (backendProcess) {
    backendProcess.kill();
    backendProcess = null;
  }

  if (process.platform !== 'darwin') {
    app.quit();
  }
});

ipcMain.handle('app:ping', async () => {
  return 'offline-notebooklm-ready';
});

ipcMain.handle('dialog:choosePath', async (_, options = {}) => {
  const result = await dialog.showOpenDialog({
    properties: ['openDirectory', 'openFile'],
    ...options,
  });
  if (result.canceled || result.filePaths.length === 0) {
    return null;
  }
  return result.filePaths[0];
});

ipcMain.handle('app:openExternal', async (_, url) => {
  if (!url) return false;
  await shell.openExternal(url);
  return true;
});

