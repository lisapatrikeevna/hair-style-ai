import { appActions } from "@/bll/app.slice";
import { AppDispatchType } from "@/bll/store";

export const handleError = (err: unknown, fallbackMessage: string, dispatch: AppDispatchType) => {
  let errorMessage = fallbackMessage;

  try {
    if (typeof err === "string") {
      errorMessage = err;
    } else if (typeof err === "object" && err !== null && "data" in err) {
      const data = (err as any).data;

      if (typeof data === "string") {
        errorMessage = data;
      } else if (typeof data?.detail === "string") {
        errorMessage = data.detail;
      } else if (typeof data?.message === "string") {
        errorMessage = data.message;
      } else if (typeof data === "object" && data !== null) {
        const parsedErrors = Object.values(data)
          .flat()
          .filter((msg) => typeof msg === "string")
          .join(" | ");

        if (parsedErrors) {
          errorMessage = parsedErrors;
        }
      }
    } else if (
      typeof err === "object" &&
      err !== null &&
      "message" in err &&
      typeof (err as any).message === "string"
    ) {
      errorMessage = (err as any).message;
    } else {
      errorMessage = JSON.stringify(err);
    }
  } catch {
    errorMessage = 'Failed to parse error';
  }

  dispatch(appActions.setError(errorMessage));
};