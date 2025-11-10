import { useEffect, useState } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import 'react-pdf/dist/Page/AnnotationLayer.css';
import 'react-pdf/dist/Page/TextLayer.css';
import './DocumentPreview.css';

// Set up PDF.js worker - use local worker for offline support
// Worker file copied from react-pdf's pdfjs-dist (version 5.4.296) to public directory
pdfjs.GlobalWorkerOptions.workerSrc = '/pdf.worker.min.mjs';

interface DocumentPreviewProps {
  isOpen: boolean;
  onClose: () => void;
  documentUrl: string;
  filename: string;
}

function DocumentPreview({ isOpen, onClose, documentUrl, filename }: DocumentPreviewProps) {
  const [numPages, setNumPages] = useState<number | null>(null);
  const [pageNumber, setPageNumber] = useState(1);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [scale, setScale] = useState(1.2);
  const [showAllPages, setShowAllPages] = useState(false);

  const isPdf = filename.toLowerCase().endsWith('.pdf');

  useEffect(() => {
    if (isOpen) {
      setPageNumber(1);
      setLoading(true);
      setError(null);
    }
  }, [isOpen, documentUrl]);

  const onDocumentLoadSuccess = ({ numPages }: { numPages: number }) => {
    setNumPages(numPages);
    setLoading(false);
    setError(null);
  };

  const onDocumentLoadError = (error: Error) => {
    setError(`Failed to load document: ${error.message}`);
    setLoading(false);
  };

  const goToPrevPage = () => {
    setPageNumber((prev) => Math.max(1, prev - 1));
  };

  const goToNextPage = () => {
    setPageNumber((prev) => Math.min(numPages || 1, prev + 1));
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      onClose();
    } else if (e.key === 'ArrowLeft') {
      goToPrevPage();
    } else if (e.key === 'ArrowRight') {
      goToNextPage();
    } else if (e.key === '+' || e.key === '=') {
      setScale((prev) => Math.min(3, prev + 0.1));
    } else if (e.key === '-') {
      setScale((prev) => Math.max(0.5, prev - 0.1));
    }
  };

  if (!isOpen) return null;

  return (
    <div className="document-preview-overlay" onClick={onClose} onKeyDown={handleKeyDown} tabIndex={-1}>
      <div className="document-preview-container" onClick={(e) => e.stopPropagation()}>
        <div className="document-preview-header">
          <h3 className="document-preview-title">{filename}</h3>
          <div className="document-preview-controls">
            {isPdf && numPages && (
              <>
                <div className="document-preview-pagination">
                  <button
                    type="button"
                    className="preview-nav-button"
                    onClick={goToPrevPage}
                    disabled={pageNumber <= 1 || showAllPages}
                  >
                    ←
                  </button>
                  <span className="preview-page-info">
                    {showAllPages ? `All pages` : `${pageNumber} / ${numPages}`}
                  </span>
                  <button
                    type="button"
                    className="preview-nav-button"
                    onClick={goToNextPage}
                    disabled={pageNumber >= numPages || showAllPages}
                  >
                    →
                  </button>
                </div>
                <button
                  type="button"
                  className="preview-view-toggle"
                  onClick={() => setShowAllPages(!showAllPages)}
                  title={showAllPages ? 'Switch to single page view' : 'Show all pages (scrollable)'}
                >
                  {showAllPages ? '📄 Single' : '📑 All'}
                </button>
              </>
            )}
            {isPdf && (
              <div className="document-preview-zoom">
                <button
                  type="button"
                  className="preview-zoom-button"
                  onClick={() => setScale((prev) => Math.max(0.5, prev - 0.1))}
                >
                  −
                </button>
                <span className="preview-zoom-info">{Math.round(scale * 100)}%</span>
                <button
                  type="button"
                  className="preview-zoom-button"
                  onClick={() => setScale((prev) => Math.min(3, prev + 0.1))}
                >
                  +
                </button>
              </div>
            )}
            <button type="button" className="preview-close-button" onClick={onClose}>
              ✕
            </button>
          </div>
        </div>

        <div className="document-preview-content">
          {loading && <div className="preview-loading">Loading document...</div>}
          {error && <div className="preview-error">{error}</div>}
          {!error && isPdf && (
            <Document
              file={documentUrl}
              onLoadSuccess={onDocumentLoadSuccess}
              onLoadError={onDocumentLoadError}
              loading={<div className="preview-loading">Loading PDF...</div>}
            >
              {showAllPages && numPages ? (
                <div className="preview-pages-container">
                  {Array.from({ length: numPages }, (_, index) => (
                    <Page
                      key={index + 1}
                      pageNumber={index + 1}
                      scale={scale}
                      renderTextLayer={true}
                      renderAnnotationLayer={true}
                      className="preview-pdf-page"
                    />
                  ))}
                </div>
              ) : (
                <div className="preview-single-page-container">
                  <Page
                    pageNumber={pageNumber}
                    scale={scale}
                    renderTextLayer={true}
                    renderAnnotationLayer={true}
                    className="preview-pdf-page"
                  />
                </div>
              )}
            </Document>
          )}
          {!error && !isPdf && (
            <iframe
              src={documentUrl}
              className="preview-iframe"
              title={filename}
              onLoad={() => setLoading(false)}
            />
          )}
        </div>
      </div>
    </div>
  );
}

export default DocumentPreview;

