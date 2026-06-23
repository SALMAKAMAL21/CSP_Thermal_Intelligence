export type TubeTemps = [number, number, number, number];

export type FormDataState = {
  video: File | null;
  tubeRef: TubeTemps;
  tubeTest: TubeTemps;
};

export type SegmentationCheck = {
  reachable: boolean;
  modelLoaded: boolean;
  backendUrl: string;
  details?: string;
};
