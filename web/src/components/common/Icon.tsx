/**
 * Material Symbols icon wrapper.
 */

interface IconProps {
  name: string;
  size?: number;
  className?: string;
  style?: React.CSSProperties;
}

export default function Icon({ name, size = 14, className = '', style }: IconProps) {
  return (
    <span
      className={`material-symbols-outlined ${className}`}
      style={{ fontSize: `${size}px`, lineHeight: 1, ...style }}
    >
      {name}
    </span>
  );
}
