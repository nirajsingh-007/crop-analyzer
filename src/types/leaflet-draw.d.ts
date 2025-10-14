declare module "react-leaflet-draw" {
  import { Map } from "leaflet";
  import { Component } from "react";

  interface EditControlProps {
    position: string;
    onCreated?: (e: any) => void;
    onEdited?: (e: any) => void;
    onDeleted?: (e: any) => void;
    draw?: any;
    edit?: any;
    map?: Map;
  }

  export class EditControl extends Component<EditControlProps> {}
}
