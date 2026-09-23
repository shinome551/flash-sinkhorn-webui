import { useMutation } from '@tanstack/react-query'
import { useCallback, useEffect, useRef, useState } from 'react'
import { uploadImage } from '../api/client.ts'
import type { ImageUploadResponse } from '../api/types.ts'
import { isAbortError } from '../lib/errors.ts'

/**
 * 画像 1 枚のアップロード。進捗 (0..1) とローカルプレビュー URL を持つ。
 * 新しいファイルを選ぶと、進行中のアップロードは中断される。
 */
export function useImageUpload(onUploaded: (image: ImageUploadResponse) => void) {
  const [progress, setProgress] = useState(0)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  const mutation = useMutation({
    mutationFn: (file: File) => {
      abortRef.current?.abort()
      const controller = new AbortController()
      abortRef.current = controller
      return uploadImage(file, setProgress, controller.signal)
    },
    onSuccess: (image) => {
      setPreviewUrl(null)
      onUploaded(image)
    },
  })
  const { mutate, reset } = mutation

  useEffect(() => {
    if (!previewUrl) return
    return () => URL.revokeObjectURL(previewUrl)
  }, [previewUrl])

  useEffect(() => () => abortRef.current?.abort(), [])

  const upload = useCallback(
    (file: File) => {
      setProgress(0)
      setPreviewUrl(URL.createObjectURL(file))
      mutate(file)
    },
    [mutate],
  )

  const cancel = useCallback(() => {
    abortRef.current?.abort()
    setPreviewUrl(null)
    reset()
  }, [reset])

  return {
    upload,
    cancel,
    isUploading: mutation.isPending,
    progress,
    previewUrl: mutation.isPending ? previewUrl : null,
    error: mutation.error && !isAbortError(mutation.error) ? mutation.error : null,
  }
}
