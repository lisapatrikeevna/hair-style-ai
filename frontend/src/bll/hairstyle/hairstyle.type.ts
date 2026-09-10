export interface UploadHairstylePhotosPayload {
  mainImage: File;
  ponytailImage?: File | null;
}

export interface HairstyleTaskResponse {
  taskId: string;
  status: 'PENDING' | 'PROCESSING' | 'SUCCESS' | 'FAILED';
  resultUrls?: string[];
  errorMessage?: string;
}