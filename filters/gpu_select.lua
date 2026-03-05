local gpu = os.getenv("GPU") or "nvidia"

local function should_keep(classes)
  local has_gpu_class = false
  local has_matching_class = false

  for _, class_name in ipairs(classes) do
    local selected_gpu = class_name:match("^gpu%-(.+)$")
    if selected_gpu then
      has_gpu_class = true
      if selected_gpu == gpu then
        has_matching_class = true
      end
    end
  end

  if has_gpu_class and not has_matching_class then
    return false
  end

  return true
end

function Div(el)
  if should_keep(el.classes) then
    return el
  end
  return {}
end
