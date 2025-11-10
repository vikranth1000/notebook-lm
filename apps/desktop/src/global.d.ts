export {};

declare global {
  interface Window {
    notebookBridge?: {
      ping: () => Promise<string>;
      choosePath: (options?: Record<string, unknown>) => Promise<string | null>;
      openExternal: (url: string) => Promise<boolean>;
    };
  }
}

