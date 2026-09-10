import { baseApi } from '../base-api';
import { UploadHairstylePhotosPayload, HairstyleTaskResponse } from './hairstyle.type';

export const hairstyleApi = baseApi.injectEndpoints({
  endpoints: (builder) => ({
    generateHairstyle: builder.mutation<HairstyleTaskResponse, UploadHairstylePhotosPayload>({
      query: ({ mainImage, ponytailImage }) => {
        const formData = new FormData();
        formData.append('main_image', mainImage);

        if (ponytailImage) {
          formData.append('ponytail_image', ponytailImage);
        }

        return {
          url: '/hairstyles/generate/',
          method: 'POST',
          body: formData,
        };
      },
      invalidatesTags: ['HairstyleTask'],
    }),
  }),
});

export const { useGenerateHairstyleMutation } = hairstyleApi;